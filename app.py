# uvicorn app:app --host 0.0.0.0 --port 8000 --reload

from fastapi.responses import FileResponse
from fastapi import FastAPI, HTTPException
import torch
import pandas as pd
import os
import re
from datetime import datetime
import asyncio
from fastapi import UploadFile, File

app = FastAPI()

# Lock para evitar condiciones de carrera en escrituras simultáneas
file_locks = {
    "EURUSD": asyncio.Lock(),
    "GBPAUD": asyncio.Lock(),
    "BTCUSD": asyncio.Lock(),
    "GBPUSD": asyncio.Lock(),
    "AUDUSD": asyncio.Lock(),
    "US30": asyncio.Lock(),
}

# Definir el modelo de trading
class TradingModel(torch.nn.Module):
    def __init__(self):
        super(TradingModel, self).__init__()
        self.fc1 = torch.nn.Linear(18, 64)
        self.fc2 = torch.nn.Linear(64, 32)
        self.fc3 = torch.nn.Linear(32, 1)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        return x

# Cargar múltiples modelos en un diccionario
models = {
    "EURUSD": TradingModel(),
    "GBPAUD": TradingModel(),
    "BTCUSD": TradingModel(),
    "GBPUSD": TradingModel(),
    "AUDUSD": TradingModel(),
    "US30": TradingModel(),
}

for symbol in models:
    model_path = os.path.join('Trading_Model', f'trading_model_{symbol}.pth')
    models[symbol].load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

for model in models.values():
    model.eval()

# Definir umbrales de dif por símbolo
dif_thresholds = {
    "EURUSD": 0.0005,
    "GBPUSD": 0.0005,
    "AUDUSD": 0.0005,
    "GBPAUD": 0.001,
    "BTCUSD": 500,
    "US30": 100,
}

# Rango de normalización con nombres cortos
min_max_dict = {
    "o5": (1.04931, 1.05013), "c5": (1.04931, 1.05013), "h5": (1.04931, 1.05013), "l5": (1.04931, 1.05013), "v5": (610, 995),
    "o15": (1.04931, 1.05013), "c15": (1.04931, 1.05013), "h15": (1.04931, 1.05013), "l15": (1.04931, 1.05013), "v15": (1461, 3005),
    "r5": (0, 100), "r15": (0, 100),
    "m5": (0, 100), "s5": (0, 100), "m15": (0, 100), "s15": (0, 100),
    "fill": (0, 1)
}

def normalize(value, min_val, max_val):
    return (value - min_val) / (max_val - min_val) if max_val != min_val else 0

@app.get("/")
def home():
    return {"message": "API funcionando correctamente"}

@app.get("/predict")
async def predict(
    symbol: str,  
    o5: float, c5: float, h5: float, l5: float, v5: float,  
    o15: float, c15: float, h15: float, l15: float, v15: float, 
    r5: float, r15: float, m5: float, s5: float, m15: float, s15: float, fill: int = 0
):
    if symbol not in models:
        raise HTTPException(status_code=400, detail=f"Modelo no disponible para {symbol}")

    data = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "symbol": symbol,
        "o5": o5, "c5": c5, "h5": h5, "l5": l5, "v5": v5,
        "o15": o15, "c15": c15, "h15": h15, "l15": l15, "v15": v15,
        "r5": r5, "r15": r15, "m5": m5, "s5": s5, "m15": m15, "s15": s15, "fill": fill,
    }

    df = pd.DataFrame([data])
    df['dif'] = abs(df['h5'] - df['l5'])

    dif_threshold = dif_thresholds[symbol]

    if df['dif'].values[0] <= dif_threshold:
        prediction = "NADA"
    else:
        for col in min_max_dict:
            df[col] = df[col].apply(lambda x: normalize(x, *min_max_dict[col]))

        input_tensor = torch.tensor(df[list(min_max_dict.keys()) + ["dif"]].values, dtype=torch.float32)
        raw_pred = models[symbol](input_tensor).item()

        if raw_pred >= 0.1:
            prediction = "BUY"
        elif raw_pred <= -0.1:
            prediction = "SELL"
        else:
            prediction = "NADA"

    df['prediction'] = prediction

    pred_path = os.path.join(os.getcwd(), 'Predictions_Files')
    os.makedirs(pred_path, exist_ok=True)
    file_name = os.path.join(pred_path, f"save_predictions_{symbol}.xlsx")

    # Bloqueo por símbolo para evitar condiciones de carrera
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
def download_file(symbol: str):
    file_path = os.path.join(os.getcwd(), 'Predictions_Files', f"save_predictions_{symbol}.xlsx")
    if os.path.exists(file_path):
        return FileResponse(path=file_path, filename=f"save_predictions_{symbol}.xlsx", media_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
    return {"error": "Archivo no encontrado"}

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    try:
        filename = file.filename
        # Extraer símbolo del nombre del archivo (ej. Data_EURUSD_2025-05-09.xlsx → EURUSD)
        match = re.search(r"([A-Z]{6})_", filename)
        if not match:
            raise HTTPException(status_code=400, detail="El nombre del archivo no contiene un par válido (AUDUSD, BTCUSD, etc.)")

        symbol = match.group(1)
        if symbol not in models:
            raise HTTPException(status_code=400, detail=f"El símbolo {symbol} no está permitido.")

        # Guardar todos en una única carpeta Uploaded_Files (sin subcarpetas por símbolo)
        upload_path = os.path.join(os.getcwd(), 'Uploaded_Files')
        os.makedirs(upload_path, exist_ok=True)
        save_path = os.path.join(upload_path, filename)

        # Guardar el archivo en disco
        with open(save_path, "wb") as f:
            content = await file.read()
            f.write(content)

        return {"message": f"✅ Archivo {filename} guardado exitosamente en Uploaded_Files/."}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"❌ Error al guardar archivo: {str(e)}")
