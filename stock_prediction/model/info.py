import datetime

from dataclasses import dataclass, asdict, field
from pydantic import BaseModel


@dataclass
class ModelInfo:
    dataset: list[float]
    train_predict: list[float]
    test_predict: list[float]
    mse_train: float
    mse_test: float
    create_date: datetime.datetime

    def __iter__(self):
        return iter(asdict(self))


@dataclass
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
        orm_model = True

    def __iter__(self):
        return iter(asdict(self))
