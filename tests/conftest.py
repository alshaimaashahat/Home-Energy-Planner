"""Shared fixtures. Data-dependent tests use the real dataset (downloaded on first use by prepare_data())."""
import pytest
from planner.data import prepare_data
from planner.nn import train_forecaster
from planner.config import SEED


@pytest.fixture(scope="session")
def data():
    return prepare_data()


@pytest.fixture(scope="session")
def forecaster(data):
    return train_forecaster(data.X_train, data.Y_train, data.X_val, data.Y_val, seed=SEED)
