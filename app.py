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

app = FastAPI()

# ======================= MODELOS Y CONFIGURACIÓN ==========================

class TradingModel18(torch.nn.Module):
    def __init__(self):
        super(TradingModel18, self).__init__()
        self.fc1 = torch.nn.Linear(18, 64)
        self.fc2 = torch.nn.Linear(64, 32)
        self.fc3 = torch.nn.Linear(32, 1)
        self.relu = torch.nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        return x

class TradingModel19(torch.nn.Module):
    def __init__(self):
        super(TradingModel19, self).__init__()
        self.fc1 = torch.nn.Linear(19, 64)
        self.fc2 = torch.nn.Linear(64, 32)
        self.fc3 = torch.nn.Linear(32, 1)
        self.relu = torch.nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        return x

forex_symbols = ["EURUSD", "GBPAUD", "BTCUSD", "GBPUSD", "AUDUSD"]
index_symbols = ["US30", "GER40", "NAS100"]
all_symbols = forex_symbols + index_symbols
# Crear un Lock por cada símbolo para evitar conflictos de escritura
file_locks = {symbol: Lock() for symbol in all_symbols}

models = {}
locks = {}
thresholds = {}

# Modelo 18 features
for sym in forex_symbols:
    model = TradingModel18()
    model.load_state_dict(torch.load(f'Trading_Model/trading_model_{sym}.pth', map_location='cpu'))
    model.eval()
    models[sym] = model
    locks[sym] = asyncio.Lock()

# Modelo 19 features
for sym in index_symbols:
    model = TradingModel19()
    model.load_state_dict(torch.load(f'Trading_Model/trading_model_{sym}.pth', map_location='cpu'))
    model.eval()
    models[sym] = model
    locks[sym] = asyncio.Lock()

# Thresholds
thresholds.update({
    "EURUSD": 0.0005,
    "GBPUSD": 0.0005,
    "AUDUSD": 0.0005,
    "GBPAUD": 0.001,
    "BTCUSD": 500,
    "US30": 50,
    "GER40": 70,
    "NAS100": 50,
})

min_max_dict = {
    "o5": (1.04931, 1.05013), "c5": (1.04931, 1.05013), "h5": (1.04931, 1.05013), "l5": (1.04931, 1.05013), "v5": (610, 995),
    "o15": (1.04931, 1.05013), "c15": (1.04931, 1.05013), "h15": (1.04931, 1.05013), "l15": (1.04931, 1.05013), "v15": (1461, 3005),
    "r5": (0, 100), "r15": (0, 100),
    "m5": (0, 100), "s5": (0, 100), "m15": (0, 100), "s15": (0, 100),
    "fill": (0, 1),
    "dif": (0, 100) 
}


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
    all_symbols = list(models.keys())
    if symbol not in all_symbols:
        raise HTTPException(status_code=400, detail=f"Modelo no disponible para {symbol}")

    is_index = symbol in ["US30", "GER40", "NAS100"]
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
    else:
        for col in min_max_dict:
            df[col] = df[col].apply(lambda x: normalize(x, *min_max_dict[col]))

        input_cols = list(min_max_dict.keys())
        input_tensor = torch.tensor(df[input_cols].values, dtype=torch.float32)

        raw_pred = model(input_tensor).item()

        if raw_pred >= 0.1:
            prediction = "BUY"
        elif raw_pred <= -0.1:
            prediction = "SELL"
        else:
            prediction = "NADA"

    df["prediction"] = prediction

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

@app.get("/download/{symbol}")
def download(symbol: str):
    path = os.path.join("Predictions_Files", f"save_predictions_{symbol}.xlsx")
    if os.path.exists(path):
        return FileResponse(path, filename=os.path.basename(path), media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    return {"error": "Archivo no encontrado"}


@app.post("/upload")
async def upload_real_prediction(file: UploadFile = File(...)):
    if not file.filename.endswith(".xlsx"):
        raise HTTPException(status_code=400, detail="Solo se permiten archivos .xlsx")

    # Validar formato del nombre del archivo
    match = re.match(r"Data_([A-Z]+)_(\d{4}-\d{2}-\d{2})\.xlsx", file.filename)
    if not match:
        raise HTTPException(status_code=400, detail="Nombre de archivo inválido. Debe ser TX_<SIMBOLO>_<YYYY-MM-DD>.xlsx")

    # Directorio de destino temporal
    dest_dir = "/tmp/Real_Predictions"
    os.makedirs(dest_dir, exist_ok=True)

    save_path = os.path.join(dest_dir, file.filename)

    try:
        with open(save_path, "wb") as f:
            contents = await file.read()
            f.write(contents)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"No se pudo guardar el archivo: {str(e)}")

    return {"message": f"Archivo {file.filename} guardado correctamente en {dest_dir}"}
