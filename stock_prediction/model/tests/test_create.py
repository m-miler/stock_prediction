import pandas as pd
import pytest

from ..info import ModelParameters, ModelInfo
from ..create import ModelCreate


@pytest.fixture
def model_params():
    params = ModelParameters(
        ticker="ALE",
        model="lstm",
        units=32,
        input_shape=(1, 3)
    )
    return params


@pytest.fixture
def model(model_params):
    return ModelCreate(model_params=model_params)




