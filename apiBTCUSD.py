# uvicorn app:app --host 0.0.0.0 --port 8000 --reload

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import FileResponse
import torch
import pandas as pd
import os
import re
from datetime import datetime
import asyncio
from asyncio import Lock
import io
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload, MediaFileUpload
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# ======================= CONFIGURACIÓN INICIAL ==========================

def get_credentials_path():
    render_secret_path = '/etc/secrets/credentials.json'
    local_path = 'credentials.json'
    if os.path.exists(render_secret_path):
        return render_secret_path
    elif os.path.exists(local_path):
        return local_path
    else:
        raise FileNotFoundError("No se encontró el archivo credentials.json ni en /etc/secrets/ ni en el directorio actual")

SERVICE_ACCOUNT_FILE = get_credentials_path()
FOLDER_ID = '1PtmUJhIpBVpvQ_FHhmxPJUkzj0wAtUQp'

creds = service_account.Credentials.from_service_account_file(
    SERVICE_ACCOUNT_FILE,
    scopes=['https://www.googleapis.com/auth/drive']
)
service = build('drive', 'v3', credentials=creds)

app = FastAPI()

# ======================= MODELOS Y NORMALIZACIÓN ==========================

class TradingModelDual(torch.nn.Module):
    def __init__(self, input_dim):
        super(TradingModelDual, self).__init__()
        self.shared = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 32),
            torch.nn.ReLU()
        )
        self.out_tipo = torch.nn.Linear(32, 1)
        self.out_profit = torch.nn.Linear(32, 1)

    def forward(self, x):
        x = self.shared(x)
        tipo = torch.tanh(self.out_tipo(x))
        profit = self.out_profit(x)
        return tipo, profit

forex_symbols = ["EURUSD", "GBPAUD", "BTCUSD", "GBPUSD", "AUDUSD"]
index_symbols = ["US30", "GER40", "NAS100"]
all_symbols = forex_symbols + index_symbols

file_locks = {symbol: Lock() for symbol in all_symbols}
models = {}
locks = {}
thresholds = {
    "EURUSD": 0.0005, "GBPUSD": 0.0005, "AUDUSD": 0.0005, "GBPAUD": 0.001,
    "BTCUSD": 500, "US30": 50, "GER40": 70, "NAS100": 50
}

min_max_dict = {
    "o5": (1.04931, 1.05013), "c5": (1.04931, 1.05013), "h5": (1.04931, 1.05013), "l5": (1.04931, 1.05013), "v5": (610, 995),
    "o15": (1.04931, 1.05013), "c15": (1.04931, 1.05013), "h15": (1.04931, 1.05013), "l15": (1.04931, 1.05013), "v15": (1461, 3005),
    "r5": (0, 100), "r15": (0, 100), "m5": (0, 100), "s5": (0, 100), "m15": (0, 100), "s15": (0, 100),
    "fill": (0, 1), "dif": (0, 100)
}

input_dim = len(min_max_dict)

for sym in forex_symbols:
    model = TradingModelDual(input_dim=input_dim)
    model.load_state_dict(torch.load(f'Trading_Model/trading_model_{sym}.pth', map_location='cpu'))
    model.eval()
    models[sym] = model
    locks[sym] = asyncio.Lock()

def normalize(value, min_val, max_val):
    return (value - min_val) / (max_val - min_val) if max_val != min_val else 0

# ======================= ENDPOINTS ==========================

@app.get("/")
def home():
    return {"message": "API combinada funcionando correctamente"}

@app.get("/predict")
async def predict(
    symbol: str,
    o5: float, c5: float, h5: float, l5: float, v5: float,
    o15: float, c15: float, h15: float, l15: float, v15: float,
    r5: float, r15: float, m5: float, s5: float, m15: float, s15: float,
    fill: int = 0
):
    if symbol not in models:
        raise HTTPException(status_code=400, detail=f"Modelo no disponible para {symbol}")

    model = models[symbol]

    data = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "symbol": symbol,
        "o5": o5, "c5": c5, "h5": h5, "l5": l5, "v5": v5,
        "o15": o15, "c15": c15, "h15": h15, "l15": l15, "v15": v15,
        "r5": r5, "r15": r15, "m5": m5, "s5": s5, "m15": m15, "s15": s15,
        "fill": fill,
    }

    df = pd.DataFrame([data])
    df["dif"] = abs(df["h5"] - df["l5"])

    if df["dif"].values[0] <= thresholds[symbol]:
        prediction = "NADA"
        tipo = 0.0
        profit = 0.0
    else:
        for col in min_max_dict:
            df[col] = df[col].apply(lambda x: normalize(x, *min_max_dict[col]))

        input_cols = list(min_max_dict.keys())
        input_tensor = torch.tensor(df[input_cols].values, dtype=torch.float32)

        raw_tipo, raw_profit = model(input_tensor)
        tipo = raw_tipo.item()
        profit = raw_profit.item()

        if tipo >= 0.1:
            prediction = "BUY"
        elif tipo <= -0.1:
            prediction = "SELL"
        else:
            prediction = "NADA"

    df["prediction"] = prediction
    df["tipo_raw"] = tipo
    df["profit_pred"] = profit

    pred_path = os.path.join(os.getcwd(), "Predictions_Files")
    os.makedirs(pred_path, exist_ok=True)
    file_name = os.path.join(pred_path, f"save_predictions_{symbol}.xlsx")

    async with file_locks[symbol]:
        try:
            if os.path.exists(file_name):
                existing_df = pd.read_excel(file_name)
                updated_df = pd.concat([existing_df, df], ignore_index=True)
            else:
                updated_df = df
            updated_df.to_excel(file_name, index=False)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error guardando predicción: {str(e)}")

    return {"symbol": symbol, "prediction": prediction}

# Otros endpoints como /upload, /download y /feedback seguirían igual con las correcciones discutidas
# ¿Quieres que también los corrija y añada aquí completos?
