import numpy as np
import pickle
import os

from stock_prediction.model.create import ModelCreate
from stock_prediction.model.info import ModelParameters


class Model:

    def __init__(self, ticker: str):
        self.ticker = ticker

    def predict(self) -> dict:
        # data = Get latest 3 price stock from stock api database
        # Get model and scaler
        # self.scaler.transform(date)
        # prediction = self.model.prediction
        # return  {'prediction': self.scaler.inverse_transform(prediction)}
        pass

    def get_model_info(self) -> dict:
        # TODO -> if model exists check the previous information in database
        # TODO -> Use postgres/sqlite3 and sqlalchemy
        # return model info or none
        pass

    def __getattr__(self, item):
        if item == "scaler" or item == "model":
            if os.path.isfile(os.path.join(os.pardir, f"models/{self.ticker}_{item}.pkl")):
                item = pickle.loads(os.path.join(os.pardir, f"models/{self.ticker}_{item}.pkl"))
                return item
            return f"{item.title()} has not been implemented yet."
        return self.item
