# uvicorn apiGBPAUD:app --host 0.0.0.0 --port 8000 --reload

import math
from fastapi import FastAPI, HTTPException
import torch
import pandas as pd
import os
from datetime import datetime
import torch.nn as nn
import pickle

app = FastAPI()

# ==================== MODELO ====================

class TradingModel(nn.Module): 
    def __init__(self):
        super(TradingModel, self).__init__()
        self.fc1 = nn.Linear(16, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc_tipo = nn.Linear(32, 1)
        self.fc_profit = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        tipo_out = self.tanh(self.fc_tipo(x))   # [-1, 1]
        profit_out = self.fc_profit(x)
        return tipo_out, profit_out


# Cargar modelo entrenado
model_path = "Trading_Model/trading_model_GBPAUD.pth"
model = TradingModel()
model.load_state_dict(torch.load(model_path, map_location="cpu"))
model.eval()

# ==================== CONFIGURACIÓN ====================

symbol = "GBPAUD"
with open("Trading_Model/min_max_GBPAUD.pkl", "rb") as f:
    min_max_dict = pickle.load(f)

def normalize(value, min_val, max_val):
    return (value - min_val) / (max_val - min_val) if max_val != min_val else 0

# ==================== ENDPOINT ====================

@app.get("/predict")
def predict(
    o5: float, c5: float, h5: float, l5: float, v5: float,
    o15: float, c15: float, h15: float, l15: float, v15: float,
    r5: float, r15: float, m5: float, s5: float, m15: float, s15: float
):
    data = {
    "precioopen5": o5, "precioclose5": c5, "preciohigh5": h5, "preciolow5": l5, "volume5": v5,
    "precioopen15": o15, "precioclose15": c15, "preciohigh15": h15, "preciolow15": l15, "volume15": v15,
    "rsi5": r5, "rsi15": r15,
    "iStochaMain5": m5, "iStochaSign5": s5,
    "iStochaMain15": m15, "iStochaSign15": s15
    }

    # Normalizar usando el mismo min_max_dict cargado desde archivo
    normalized = []
    for key in data:
        min_val, max_val = min_max_dict[key]
        normalized.append(normalize(data[key], min_val, max_val))

    input_tensor = torch.tensor([normalized], dtype=torch.float32)

    with torch.no_grad():
        tipo_pred_tensor, profit_pred_tensor = model(input_tensor)
        tipo_pred = tipo_pred_tensor.item()
        profit_pred = profit_pred_tensor.item()

    # Desnormalizar profit
    min_profit, max_profit = min_max_dict['profit']
    profit_pred = profit_pred * (max_profit - min_profit) + min_profit

    if tipo_pred >= 0.1:
        prediction = "BUY"
    elif tipo_pred <= -0.1:
        prediction = "SELL"
    else:
        prediction = "NADA"

    if not math.isfinite(profit_pred):
        profit_pred = 0.0

    return {
        "symbol": symbol,
        "timestamp": datetime.now().isoformat(),
        "prediction": prediction,
        "profit": round(profit_pred, 6) if prediction != "NADA" else None
    }

