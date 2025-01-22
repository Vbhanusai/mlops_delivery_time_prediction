import numpy as np
import pandas as pd

from zenml import step, pipeline
from zenml.config import DockerSettings
from zenml.constants import DEFAULT_SERVICE_START_STOP_TIMEOUT

from zenml.integrations.constants import MLFLOW
from zenml.integrations.mlflow import services
from zenml.integrations.mlflow.model_deployers.mlflow_model_deployer import (
    MLFlowModelDeployer,
)

from zenml.integrations.mlflow.services import MLFlowDeploymentService

from zenml.integrations.mlflow.steps import mlflow_model_deployer_step
import json
import platform

from steps.clean_data import clean_data
from steps.ingest_data import ingest_data
from steps.model_train import model_train
from steps.evaluation import evaluate_model

from .utils import get_data_for_test

docker_settings = DockerSettings(required_integrations=[MLFLOW])

class DeplymentTriggerConfig:
    min_accuracy = 0.3

@step(enable_cache=False)
def dynamic_importer()-> str:
    data = get_data_for_test()
    return data

@step
def deployment_trigger(
    accuracy:float):
    """
    Trigger deployment based on accuracy
    """
    config = DeplymentTriggerConfig
    # return accuracy >= config.min_accuracy
    return True

class MLFlowDeploymentLoaderStepParameters():
    """MLflow deployment getter parameters

    Attributes:
        pipeline_name: name of the pipeline that deployed the MLflow prediction
            server
        step_name: the name of the step that deployed the MLflow prediction
            server
        running: when this flag is set, the step only returns a running service
        model_name: the name of the model that is deployed
    """

    pipeline_name: str
    step_name: str
    running: bool = True

@step(enable_cache=False)
def prediction_service_loader(
    pipeline_name: str,
    pipeline_step_name: str,
    running: bool = True,
    model_name: str = "model",
) -> MLFlowDeploymentService:
    """Get the prediction service started by the deployment pipeline"""
    mlflow_model_deployer_component = MLFlowModelDeployer.get_active_model_deployer()
    existing_services = mlflow_model_deployer_component.find_model_server(
        pipeline_name=pipeline_name,
        pipeline_step_name=pipeline_step_name,
        model_name=model_name,
        running=running,
    )

    if not existing_services:
        raise RuntimeError(
            f"No MLflow prediction service deployed by the "
            f"{pipeline_step_name} step in the {pipeline_name} "
            f"pipeline for the '{model_name}' model is currently "
            f"running."
        )

    service = existing_services[0]
    if not service.is_running:
        service.start(timeout=60)
    return service

@step
def predictor(
    service: MLFlowDeploymentService,
    data: str,
) -> np.ndarray:
    """Run an inference request against a prediction service"""

    service.start(timeout=10)  # should be a NOP if already started
    data = json.loads(data)
    data.pop("columns")
    data.pop("index")
    columns_for_df = [
        'Distance_km', 'Preparation_Time_min', 'Courier_Experience_yrs', 'Weather_Foggy', 'Weather_Rainy', 'Weather_Snowy',
       'Weather_Windy', 'Traffic_Level_Low', 'Traffic_Level_Medium',
       'Time_of_Day_Evening', 'Time_of_Day_Morning', 'Time_of_Day_Night',
       'Vehicle_Type_Car', 'Vehicle_Type_Scooter'
    ]
    
    df = pd.DataFrame(data["data"], columns=columns_for_df)
    json_list = json.loads(json.dumps(list(df.T.to_dict().values())))
    data = np.array(json_list)
    prediction = service.predict(data)
    return prediction



@pipeline(enable_cache=False, settings={"docker":docker_settings})
def continuous_deployment_pipeline(
    min_accuracy:float=0.3,
    workers:int = 1,
    timeout:int = DEFAULT_SERVICE_START_STOP_TIMEOUT
    ):

    data_path="data/Food_Delivery_Times.csv"
    df = ingest_data(data_path)
    X_train, X_test, y_train, y_test = clean_data(df)
    model = model_train(X_train, y_train)
    mse, r2, mae = evaluate_model(model, X_test, y_test)
    
    deployment_decision = deployment_trigger(accuracy=r2)
    mlflow_model_deployer_step(
        model=model,
        deploy_decision=deployment_decision,
        workers=workers,
        timeout=timeout
    )


@pipeline(enable_cache=False, settings={"docker":docker_settings})
def inference_pipeline(
    pipeline_name: str,
    pipeline_step_name: str
    ):
    batch_data = dynamic_importer()
    model_deployment_service = prediction_service_loader(
        pipeline_name=pipeline_name,
        pipeline_step_name=pipeline_step_name,
        running=False,
    )
    prediction = predictor(service=model_deployment_service, data=batch_data)
    return prediction
