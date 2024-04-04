import datetime

import numpy as np

from dataclasses import dataclass, asdict


@dataclass
class ModelInfo:
    dataset: np.array
    train_predict: np.array
    test_predict: np.array
    mse_train: float
    mse_test: float
    create_date: datetime.datetime

    def save_to_db(self):
        pass


@dataclass
class ModelParameters:
    ticker: str
    model: str
    units: int
    input_shape: tuple
    epochs: int = 50
    batch_size: int = 1
    activation: str = 'relu'
    optimizer: str = 'adam'

    def __iter__(self):
        return iter(asdict(self))

    def save_to_db(self):
        pass
