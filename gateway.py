# uvicorn gateway:app --host 0.0.0.0 --port 8000 --reload

from fastapi import FastAPI, Request, HTTPException
import httpx

app = FastAPI()

# Definimos listas de símbolos por backend
api_forex = ["EURUSD", "GBPUSD", "AUDUSD", "GBPAUD", "BTCUSD"]
api_indicator = ["US30", "GER40", "NAS100"]

# Creamos el diccionario final combinando las listas
symbol_routing = {
    symbol: "http://localhost:8001" for symbol in api_forex
} | {
    symbol: "http://localhost:8002" for symbol in api_indicator
}


@app.get("/")
def home():
    return {"message": "Gateway funcionando correctamente"}

@app.get("/predict")
async def gateway_predict(request: Request):
    params = dict(request.query_params)
    symbol = params.get("symbol")

    if not symbol or symbol not in symbol_routing:
        raise HTTPException(status_code=400, detail="Símbolo inválido o no soportado")

    backend_url = f"{symbol_routing[symbol]}/predict"

    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(backend_url, params=params)
            return response.json()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error al conectar con el backend: {str(e)}")

@app.get("/download/{symbol}")
async def gateway_download(symbol: str):
    if symbol not in symbol_routing:
        raise HTTPException(status_code=400, detail="Símbolo inválido o no soportado")

    backend_url = f"{symbol_routing[symbol]}/download/{symbol}"

    async with httpx.AsyncClient() as client:
        response = await client.get(backend_url)
        if response.status_code == 200:
            return response
        else:
            raise HTTPException(status_code=response.status_code, detail=response.text)
