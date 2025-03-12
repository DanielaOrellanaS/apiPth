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
    "po5": (1.04931, 1.05013), "pc5": (1.04931, 1.05013), "ph5": (1.04931, 1.05013), "pl5": (1.04931, 1.05013), "v5": (610, 995),
    "po15": (1.04931, 1.05013), "pc15": (1.04931, 1.05013), "ph15": (1.04931, 1.05013), "pl15": (1.04931, 1.05013), "v15": (1461, 3005),
    "r5": (0, 100), "r15": (0, 100),
    "sm5": (0, 100), "ss5": (0, 100), "sm15": (0, 100), "ss15": (0, 100),
    "fill": (0, 1)
}

def normalize(value, min_val, max_val):
    return (value - min_val) / (max_val - min_val) if max_val != min_val else 0

@app.get("/")
def home():
    return {"message": "API funcionando correctamente"}

@app.get("/predict")
def predict(
    po5: float, pc5: float, ph5: float, pl5: float, v5: int,
    po15: float, pc15: float, ph15: float, pl15: float, v15: int,
    r5: float, r15: float, sm5: float, ss5: float, sm15: float, ss15: float, fill: int
):
    """
    Recibe datos a través de la URL, los normaliza y hace una predicción.
    """

    # Crear el diccionario de datos con los valores recibidos
    data = {
        "po5": po5, "pc5": pc5, "ph5": ph5, "pl5": pl5, "v5": v5,
        "po15": po15, "pc15": pc15, "ph15": ph15, "pl15": pl15, "v15": v15,
        "r5": r5, "r15": r15, "sm5": sm5, "ss5": ss5, "sm15": sm15, "ss15": ss15, "fill": fill
    }

    df = pd.DataFrame([data])

    # Calcular la diferencia "dif"
    df['dif'] = abs(df['ph5'] - df['pl5'])

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
