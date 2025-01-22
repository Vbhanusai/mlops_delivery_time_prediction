import logging

from typing import Tuple
from typing_extensions import Annotated
from zenml import step
import pandas as pd
from src.evaluation_metrics import MSE, R2,MAE
from sklearn.base import RegressorMixin
import mlflow
import pandas as pd
from zenml.client import Client

experiment_tracker = Client().active_stack.experiment_tracker
@step(experiment_tracker=experiment_tracker.name)
def evaluate_model(model,
    X_test:pd.DataFrame,
    y_test:pd.Series)-> Tuple[
        Annotated[float, "MSE"],
        Annotated[float, "R2"],
        Annotated[float, "MAE"]
    ]:
    try:
        # Ensure MLflow experiment is created
        experiment_name = "training_pipeline"
        mlflow.set_experiment(experiment_name)

        predictions = model.predict(X_test)
        y_test = y_test.to_numpy()
        temp =predictions
        temp = pd.DataFrame(predictions)

        mse_class = MSE()
        mse = mse_class.calculate_scores(y_test, predictions)
        mlflow.log_metric("MSE", mse)

        r2_class = R2()
        r2 = r2_class.calculate_scores(y_test, predictions)
        mlflow.log_metric("R2", r2)

        mae_class = MAE()
        mae = mae_class.calculate_scores(y_test, predictions)
        mlflow.log_metric("MAE", mae)
        
        return mse, r2, mae
    except Exception as e:
        logging.error("Error in evaluating model: {}".format(e))
        return e