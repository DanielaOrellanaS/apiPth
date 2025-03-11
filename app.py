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

# Rango de normalización (ajusta estos valores según tu dataset real)
min_max_dict = {
    "precioopen5": (1.04931, 1.05013),
    "precioclose5": (1.04931, 1.05013),
    "preciohigh5": (1.04931, 1.05013),
    "preciolow5": (1.04931, 1.05013),
    "volume5": (610, 995),
    "precioopen15": (1.04931, 1.05013),
    "precioclose15": (1.04931, 1.05013),
    "preciohigh15": (1.04931, 1.05013),
    "preciolow15": (1.04931, 1.05013),
    "volume15": (1461, 3005),
    "rsi5": (0, 100),
    "rsi15": (0, 100),
    "iStochaMain5": (0, 100),
    "iStochaSign5": (0, 100),
    "iStochaMain15": (0, 100),
    "iStochaSign15": (0, 100),
    "fill": (0, 1)
}

def normalize(value, min_val, max_val):
    return (value - min_val) / (max_val - min_val) if max_val != min_val else 0

@app.get("/predict")
def predict(
    precioopen5: float, precioclose5: float, preciohigh5: float, preciolow5: float, volume5: int,
    precioopen15: float, precioclose15: float, preciohigh15: float, preciolow15: float, volume15: int,
    rsi5: float, rsi15: float, iStochaMain5: float, iStochaSign5: float, 
    iStochaMain15: float, iStochaSign15: float, fill: int
):
    """
    Recibe datos a través de la URL, los normaliza y hace una predicción.
    """

    # Crear el diccionario de datos con los valores recibidos
    data = {
        "precioopen5": precioopen5, "precioclose5": precioclose5, "preciohigh5": preciohigh5, "preciolow5": preciolow5, "volume5": volume5,
        "precioopen15": precioopen15, "precioclose15": precioclose15, "preciohigh15": preciohigh15, "preciolow15": preciolow15, "volume15": volume15,
        "rsi5": rsi5, "rsi15": rsi15, "iStochaMain5": iStochaMain5, "iStochaSign5": iStochaSign5, 
        "iStochaMain15": iStochaMain15, "iStochaSign15": iStochaSign15, "fill": fill
    }

    df = pd.DataFrame([data])

    # Calcular la diferencia "dif"
    df['dif'] = abs(df['preciohigh5'] - df['preciolow5'])

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
