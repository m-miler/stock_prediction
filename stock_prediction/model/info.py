import datetime

from dataclasses import field
from pydantic import BaseModel


class ModelInfo(BaseModel):
    dataset: list[list[float]]
    train_predict: list[list[float]]
    test_predict: list[list[float]]
    mse_train: float
    mse_test: float
    create_date: datetime.datetime

    class Config:
        orm_mode = True


class ModelParameters(BaseModel):
    ticker: str
    model: str
    units: int
    input_shape_min: int
    input_shape_max: int
    epochs: int = 50
    batch_size: int = 1
    activation: str = 'relu'
    optimizer: str = 'adam'

    info: list[ModelInfo] = field(default_factory=list)

    class Config:
        orm_mode = True
