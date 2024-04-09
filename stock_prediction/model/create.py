import datetime

import requests
import pandas as pd
import numpy as np
import os
import pickle

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

from .base import BaseModels
from .info import ModelParameters, ModelInfo


class ModelCreate:
    base_url = 'http://localhost:8080'

    def __init__(self, model_params: ModelParameters = None):
        self.parameters = model_params
        self.scaler = MinMaxScaler(feature_range=(0, 1))

    def _get_input_data(self) -> dict:
        url = f'{self.base_url}/prices/?company_abbreviation={self.parameters.ticker}'
        response = requests.get(url)
        return response.json()

    def _preprocess_dataset(self):
        raw_data: dict = self._get_input_data()
        input_data = pd.DataFrame(data=raw_data,
                                  columns=["date", "open_price", "max_price", "min_price", "close_price"])
        dataset = input_data['close_price'].values.astype('float32').reshape(-1, 1)
        dataset = self.scaler.fit_transform(dataset)
        return dataset

    @staticmethod
    def _train_test_split(dataset: np.array):
        train_size = int(len(dataset) * 0.67)
        test_size = len(dataset) - train_size
        train, test = dataset[0:train_size, :], dataset[test_size:len(dataset), :]
        return train, test

    @staticmethod
    def _create_dataset(dataset, look_back=1):
        data_x, data_y = [], []
        for i in range(len(dataset) - look_back - 1):
            a = dataset[i:(i + look_back), 0]
            data_x.append(a)
            data_y.append(dataset[i + look_back, 0])
        return np.array(data_x), np.array(data_y)

    def _prepare_train_test_data(self, data):
        train, test = self._train_test_split(data)
        train_x, train_y = self._create_dataset(train, 3)
        test_x, test_y = self._create_dataset(test, 3)
        train_x = np.reshape(train_x, (train_x.shape[0], 1, train_x.shape[1]))
        test_x = np.reshape(test_x, (test_x.shape[0], 1, test_x.shape[1]))

        return train_x, test_x, train_y, test_y

    def create(self) -> ModelInfo:
        model = BaseModels.get_base_model(self.parameters)
        dataset = self._preprocess_dataset()
        train_x, test_x, train_y, test_y = self._prepare_train_test_data(dataset)

        model.compile(loss='mean_squared_error', optimizer=self.parameters.optimizer)
        model.fit(train_x, train_y,
                  epochs=self.parameters.epochs,
                  batch_size=self.parameters.batch_size,
                  verbose=0)

        train_predict = model.predict(train_x)
        test_predict = model.predict(test_x)

        train_predict = self.scaler.inverse_transform(train_predict)
        train_y = self.scaler.inverse_transform([train_y])
        test_predict = self.scaler.inverse_transform(test_predict)
        test_y = self.scaler.inverse_transform([test_y])

        model_info = ModelInfo(
            dataset=dataset.tolist(),
            train_predict=train_predict.tolist(),
            test_predict=test_predict.tolist(),
            mse_train=np.sqrt(mean_squared_error(train_y[0], train_predict[:, 0])),
            mse_test=np.sqrt(mean_squared_error(test_y[0], test_predict[:, 0])),
            create_date=datetime.datetime.now()
        )

        model_info.save_to_db()
        self.parameters.save_to_db()

        model.save(os.path.join(os.getcwd(), fr"models\{self.parameters.ticker}_model.h5"))

        with open(os.path.join(os.path.join(os.getcwd(),
                               fr"models\{self.parameters.ticker}_scaler.pkl")), "w+b") as file:
            pickle.dump(self.scaler, file)

        return model_info
