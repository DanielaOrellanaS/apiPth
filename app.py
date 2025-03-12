from fastapi import FastAPI
import torch
import pandas as pd

app = FastAPI()

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

# Cargar el modelo entrenado
model = TradingModel()
model.load_state_dict(torch.load('trading_model.pth'))
model.eval()

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
def predict(
    o5: float, c5: float, h5: float, l5: float, v5: int,
    o15: float, c15: float, h15: float, l15: float, v15: int,
    r5: float, r15: float, m5: float, s5: float, m15: float, s15: float, fill: int = 0  # Default fill is 0
):
    """
    Recibe datos a través de la URL, los normaliza y hace una predicción.
    """

    # Crear el diccionario de datos con los valores recibidos
    data = {
        "o5": o5, "c5": c5, "h5": h5, "l5": l5, "v5": v5,
        "o15": o15, "c15": c15, "h15": h15, "l15": l15, "v15": v15,
        "r5": r5, "r15": r15, "m5": m5, "s5": s5, "m15": m15, "s15": s15, "fill": fill
    }

    df = pd.DataFrame([data])

    # Calcular la diferencia "dif"
    df['dif'] = abs(df['h5'] - df['l5'])

    # Si "dif" es menor a 0.0005, descartar el dato
    if df['dif'].values[0] <= 0.0005:
        return {"error": "Diferencia muy pequeña, dato descartado."}

    # Normalizar los valores
    for col in min_max_dict:
        df[col] = df[col].apply(lambda x: normalize(x, *min_max_dict[col]))

    # Convertir a tensor
    input_tensor = torch.tensor(df[list(min_max_dict.keys()) + ["dif"]].values, dtype=torch.float32)

    # Hacer la predicción
    prediction = model(input_tensor).item()

    # Clasificación de la predicción
    if prediction >= 0.1:
        return {"prediction": "BUY"}
    elif prediction <= -0.1:
        return {"prediction": "SELL"}
    else:
        return {"prediction": "NEUTRAL"}
