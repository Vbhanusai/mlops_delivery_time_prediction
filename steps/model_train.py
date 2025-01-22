import logging

import pandas as pd
from zenml import step

from src.model_development import LinearRegressionModel
import mlflow
# from sklearn.base import RegressorMixin
from zenml.client import Client
from .config import ModelNameConfig

from materializer.LinearRegressionMaterializer import LinearRegressionModelMaterializer

experiment_tracker = Client().active_stack.experiment_tracker

@step(experiment_tracker=experiment_tracker.name, output_materializers=LinearRegressionModelMaterializer)
def model_train(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame) :
    
    try:
        model = None
        config = ModelNameConfig
        if config.model_name == "LinearRegression":
            mlflow.sklearn.autolog()
            model = LinearRegressionModel()
            model.train(X_train, y_train)
            return model
        else:
            raise ValueError("Model {} not supported.".format(config.model_name))
    except Exception as e:
        logging.error("Error in training model: {}".format(e))
        return e
