import numpy as np
import pickle
import os
import requests

from keras.models import load_model
from pandas.tseries.offsets import BDay
from datetime import date, timedelta


class Model:
    base_url = 'http://localhost:8080'

    def __init__(self, ticker: str):
        self.ticker = ticker

    def predict(self) -> dict:

        try:
            model = self.model
            scaler = self.scaler
            dates = [(date.today() - BDay(i)).strftime("%Y-%m-%d") for i in range(1, 4)]
            data = self._get_predict_data(dates)

        except (NotImplementedError, IndexError) as err:
            return {'error': err.args}

        data = scaler.transform(data.reshape(-1, 1)).reshape(1, -1)
        data = np.reshape(data, (data.shape[0], 1, data.shape[1]))
        prediction = model.predict(data)
        prediction = scaler.inverse_transform(prediction)
        return {'prediction': float(prediction[0][0])}

    def _get_predict_data(self, dates: list) -> np.array:
        input_data = []
        for d in dates:
            url = f'{self.base_url}/prices/?date={d}&company_abbreviation={self.ticker}'
            response = requests.get(url).json()

            try:
                input_data.append(response[0]['close_price'])
            except IndexError:
                raise IndexError(f"Missing price data for date: {d}")

        return np.array(input_data)

    def get_model_info(self) -> dict:
        # TODO -> if model exists check the previous information in database
        # TODO -> Use postgres/sqlite3 and sqlalchemy
        # return model info or none
        pass

    def __getattr__(self, item):
        if item == "scaler":
            if os.path.isfile(os.path.join(os.path.join(os.getcwd(), fr"models\{self.ticker}_scaler.pkl"))):
                item = pickle.load(
                    open(os.path.join(os.path.join(os.getcwd(), fr"models\{self.ticker}_scaler.pkl")), 'rb')
                )
                return item
            raise NotImplementedError(f"{item.title()} for {self.ticker} has not been created yet.")
        elif item == "model":
            if os.path.isfile(os.path.join(os.getcwd(), fr"models\{self.ticker}_model.h5")):
                item = load_model(os.path.join(os.getcwd(), fr"models\{self.ticker}_model.h5"))
                return item
            raise NotImplementedError(f"{item.title()} for {self.ticker} has not been created yet.")

        return self.item
