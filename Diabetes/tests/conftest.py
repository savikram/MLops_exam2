
import pytest
from sklearn.model_selection import train_test_split

from Diabetes_model.config.core import config
from Diabetes_model.processing.data_manager import load_dataset



@pytest.fixture(autouse=True, scope='session')
def sample_input_data():
    """Loads test data and returns dataframe"""
    test_data = load_dataset(file_name=config.app_config.test_data_file)

    X_test = test_data[config.models_cfg.features]
    y_test = test_data[config.models_cfg.target]
    # divide train and test
    #X_train, X_test, y_train, y_test = train_test_split(
     #   data,  # predictors
      #  data[config.models_cfg.target],
      #  test_size=config.models_cfg.test_size,
        # we are setting the random seed here
        # for reproducibility
      #  random_state=config.models_cfg.random_state,
    #)

    return X_test, y_test