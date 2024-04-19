import uvicorn
import orm.crud as crud
import sys

from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy.orm import Session

from model.model import Model
from model.create import ModelCreate
from model.info import ModelParameters, ModelInfo
from orm.models import Base
from config import settings

from dataclasses import asdict

sys.path.append("..")
Base.metadata.create_all(bind=settings.engine)

app = FastAPI()


def get_db():
    db = settings.SessionLocal()
    try:
        yield db
    finally:
        db.close()


@app.post("/model/create/", response_model=ModelInfo)
def create_model(model_params: ModelParameters):
    model: ModelInfo = ModelCreate(model_params=model_params).create()
    return model


@app.get("/model/predict/{ticker}")
def predict(ticker):
    prediction = Model(ticker=ticker).predict()
    return prediction


@app.get("/model/info/{ticker}/{model}", response_model=ModelParameters)
def model_into(ticker: str, model: str, db: Session = Depends(get_db)):
    model_info = crud.get_model_info(db, ticker=ticker, model=model)
    if model_info is None:
        raise HTTPException(status_code=404, detail="Model has not been created.")
    return model_info
