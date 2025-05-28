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
from googleapiclient.http import MediaIoBaseDownload
import torch.nn as nn   
from torch.utils.data import TensorDataset, DataLoader
from googleapiclient.http import MediaFileUpload


SERVICE_ACCOUNT_FILE = 'credentials.json'
FOLDER_ID = '1PtmUJhIpBVpvQ_FHhmxPJUkzj0wAtUQp'

creds = service_account.Credentials.from_service_account_file(
    SERVICE_ACCOUNT_FILE,
    scopes=['https://www.googleapis.com/auth/drive']
)
service = build('drive', 'v3', credentials=creds)

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

@app.post("/upload/{symbol}")
def upload_prediction_file(symbol: str):
    try:
        # Ruta donde se guarda el archivo en Render
        file_path = os.path.join("Predictions_Files", f"save_predictions_{symbol}.xlsx")
        
        if not os.path.exists(file_path):
            return {"status": "error", "message": f"No se encontró el archivo {file_path}"}
        
        # Nombre final en Google Drive
        fecha_actual = datetime.now().strftime("%Y-%m-%d")
        file_name = f"save_predictions_{symbol}.xlsx"
        
        # Crear objeto de subida
        media = MediaFileUpload(file_path, mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')

        # Subir a Google Drive
        file_metadata = {
            'name': file_name,
            'parents': [FOLDER_ID]
        }
        uploaded_file = service.files().create(
            body=file_metadata,
            media_body=media,
            fields='id'
        ).execute()

        return {
            "status": "success",
            "message": f"Archivo {file_name} subido correctamente a Drive",
            "file_id": uploaded_file.get("id")
        }

    except Exception as e:
        return {"status": "error", "message": str(e)}


def find_drive_file(symbol: str, target_date: str):
    query = f"'{FOLDER_ID}' in parents and name contains 'TX_{symbol}_{target_date}'"
    results = service.files().list(q=query, fields="files(id, name)").execute()
    files = results.get('files', [])
    return files[0] if files else None

def download_drive_file(file_id: str, destination_path: str):
    request = service.files().get_media(fileId=file_id)
    fh = io.FileIO(destination_path, 'wb')
    downloader = MediaIoBaseDownload(fh, request)
    done = False
    while not done:
        status, done = downloader.next_chunk()

class TradingMLP(nn.Module):
    def __init__(self, input_dim):
        super(TradingMLP, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Tanh()
        )

    def forward(self, x):
        return self.fc(x)

def get_model_path(symbol: str) -> str:
    base_path = r"C:\Users\user\OneDrive\Documentos\Trading\ModelPth\apiPth\Trading_Model"
    return os.path.join(base_path, f"trading_model_{symbol}.pth")

def normalize_inputs(df, feature_cols):
    # Aquí usa tu método de normalización actual, por ahora lo mantengo como min-max 0-1
    for col in feature_cols:
        min_val = df[col].min()
        max_val = df[col].max()
        if min_val != max_val:
            df[col] = (df[col] - min_val) / (max_val - min_val)
        else:
            df[col] = 0
    return df

@app.get("/feedback/{symbol}")
def feedback(symbol: str):
    today = datetime.now().strftime('%Y-%m-%d')
    pred_filename = f"save_predictions_{symbol}.xlsx"
    real_filename = f"Data_{symbol}_{today}.xlsx"

    # Función auxiliar para descargar desde Drive
    def download_from_drive(file_name: str, local_path: str):
        query = f"'{FOLDER_ID}' in parents and name = '{file_name}'"
        results = service.files().list(q=query, fields="files(id, name)").execute()
        files = results.get('files', [])
        if not files:
            raise HTTPException(status_code=404, detail=f"No se encontró {file_name} en Google Drive")
        file_id = files[0]['id']
        request = service.files().get_media(fileId=file_id)
        fh = io.FileIO(local_path, 'wb')
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while not done:
            _, done = downloader.next_chunk()

    # Descargar ambos archivos
    pred_local = f"temp_{pred_filename}"
    real_local = f"temp_{real_filename}"
    download_from_drive(pred_filename, pred_local)
    download_from_drive(real_filename, real_local)

    # Leer archivos
    try:
        pred_df = pd.read_excel(pred_local)
        real_df = pd.read_excel(real_local)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al leer archivos: {e}")

    # Preparar y comparar
    real_df = real_df.rename(columns={"simbolo": "symbol"})
    merged = pd.merge(pred_df, real_df, on="symbol", suffixes=("_pred", "_real"))

    if 'tipo' not in merged.columns or 'prediction' not in merged.columns:
        raise HTTPException(status_code=400, detail="Faltan columnas clave para feedback.")

    merged = merged.dropna(subset=['prediction', 'tipo'])
    merged = merged[merged['prediction'].round(1) != merged['tipo'].round(1)]

    if merged.empty:
        return {
            "symbol": symbol,
            "date": today,
            "total_predictions": len(pred_df),
            "incorrect_predictions": 0,
            "accuracy": 100.0,
            "retrained": False
        }

    # Reentrenamiento
    feature_cols = [col for col in pred_df.columns if col not in ['timestamp', 'symbol', 'prediction', 'dif']]
    merged = normalize_inputs(merged, feature_cols)

    X = merged[feature_cols].values.astype('float32')
    y = merged['tipo'].values.astype('float32').reshape(-1, 1)

    model_path = get_model_path(symbol)
    model = TradingMLP(input_dim=len(feature_cols))
    model.load_state_dict(torch.load(model_path))
    model.train()

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    dataset = TensorDataset(torch.tensor(X), torch.tensor(y))
    loader = DataLoader(dataset, batch_size=16, shuffle=True)

    for epoch in range(10):
        for xb, yb in loader:
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()

    torch.save(model.state_dict(), model_path)

    # Limpiar temporales
    os.remove(pred_local)
    os.remove(real_local)

    return {
        "symbol": symbol,
        "date": today,
        "total_predictions": len(pred_df),
        "incorrect_predictions": len(merged),
        "accuracy": round(100 * (1 - len(merged) / len(pred_df)), 2),
        "retrained": True
    }