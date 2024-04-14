from sqlalchemy.orm import Session
from .models import ModelInfoDb, ModelParametersDb


def get_model_info(db: Session, ticker: str, model: str):
    return db.query(ModelParametersDb).filter(ModelParametersDb.id == f'{ticker}_{model}').first()
