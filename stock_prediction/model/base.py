from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from .info import ModelParameters


class BaseModels:

    @staticmethod
    def get_base_model(model_param: ModelParameters):
        match model_param.model:
            case 'lstm':
                return BaseModels.lstm_base_model(model_param=model_param)
            case _:
                raise f"Chosen model is not available. Please select another one!"

    @staticmethod
    def lstm_base_model(model_param: ModelParameters):

        model = Sequential()
        # TODO -> consider to add a transfer learning
        model.add(LSTM(units=model_param.units,
                       input_shape=(model_param.input_shape_min, model_param.input_shape_max),
                       activation=model_param.activation))
        model.add(Dense(1))
        return model
