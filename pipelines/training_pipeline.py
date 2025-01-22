from zenml import pipeline
from steps.ingest_data import ingest_data
from steps.clean_data import clean_data
from steps.model_train import model_train
from steps.evaluation import evaluate_model

from zenml.config import DockerSettings
from zenml.integrations.constants import MLFLOW

docker_settings = DockerSettings(required_integrations=[MLFLOW])

@pipeline(enable_cache=True,settings={"docker": docker_settings})
def training_pipeline(data_path: str):
    df = ingest_data(data_path)
    X_train, X_test, y_train, y_test = clean_data(df)
    model=model_train(X_train, y_train)
    mse, r2, mae = evaluate_model(model, X_test, y_test)