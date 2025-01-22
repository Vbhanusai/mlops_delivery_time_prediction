import os
import pickle
from typing import Type

from zenml.materializers.base_materializer import BaseMaterializer
from zenml.enums import ArtifactType
from zenml.io import fileio
from src.model_development import LinearRegressionModel  # Import your custom class

DEFAULT_FILENAME = "LinearRegressionModel.pkl"


class LinearRegressionModelMaterializer(BaseMaterializer):
    """
    Custom materializer for LinearRegressionModel artifacts.
    """

    ASSOCIATED_TYPES = (LinearRegressionModel,)  # Link to your custom model class
    ASSOCIATED_ARTIFACT_TYPE = ArtifactType.MODEL  # Use the generic MODEL artifact type

    def load(self, data_type: Type[LinearRegressionModel]) -> LinearRegressionModel:
        """
        Deserialize the LinearRegressionModel from the artifact store.

        Args:
            data_type: The type of data being deserialized.

        Returns:
            The deserialized LinearRegressionModel object.
        """
        filepath = os.path.join(self.uri, DEFAULT_FILENAME)
        with fileio.open(filepath, "rb") as f:
            model = pickle.load(f)
        return model

    def save(self, model: LinearRegressionModel) -> None:
        """
        Serialize the LinearRegressionModel to the artifact store.

        Args:
            model: The LinearRegressionModel object to serialize.
        """
        filepath = os.path.join(self.uri, DEFAULT_FILENAME)
        with fileio.open(filepath, "wb") as f:
            pickle.dump(model, f)

    def extract_metadata(self, model: LinearRegressionModel) -> dict:
        """
        Extract metadata about the LinearRegressionModel.

        Args:
            model: The LinearRegressionModel object.

        Returns:
            Metadata dictionary containing model details.
        """
        return {
            "model_type": type(model).__name__,
            "parameters": getattr(model.model, "get_params", lambda: None)(),
        }
