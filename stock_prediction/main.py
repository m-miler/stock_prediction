import uvicorn
from typing import Union
from fastapi import FastAPI

from model.model import Model
from model.create import ModelCreate
from model.info import ModelParameters, ModelInfo

from dataclasses import asdict

app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/model/create/")
async def create_model(model_params: ModelParameters):
    model: ModelInfo = ModelCreate(model_params=model_params).create()
    return asdict(model)


@app.get("/model/predict/{ticker}")
def predict(ticker):
    prediction = Model(ticker=ticker).predict()
    return prediction


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
