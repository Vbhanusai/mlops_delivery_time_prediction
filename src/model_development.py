import logging
from abc import ABC, abstractmethod
from sklearn.linear_model import LinearRegression
class Model(ABC):
    """
    Abstract class for models.
    """
    @abstractmethod
    def train(self, X_train, y_train):
        pass

    @abstractmethod
    def predict(self, X_test):
        pass

class LinearRegressionModel(Model):
    """
    Concrete class for linear regression model.
    """
    def __init__(self):
        self.model = LinearRegression()

    def train(self, X_train, y_train, **kwargs):
        """
        Trains the model.
        Args:
            X_train: pd.DataFrame: Training features.
            y_train: pd.Series: Training target.
        """
        try:
            self.model.fit(X_train, y_train)
            logging.info("Model training completed.")
        except:
            logging.error("Error in training model.") 

    def predict(self, X_test):
        """
        Predicts the target.
        Args:
            X_test: pd.DataFrame: Testing features.
        Returns:
            pd.Series: Predicted target.
        """
        y_pred = self.model.predict(X_test)
        return y_pred