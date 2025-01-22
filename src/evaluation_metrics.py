import logging
from abc import ABC, abstractmethod

import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

class EvaluationStrategy(ABC):
    """
    Abstract class for evaluation strategies.
    """
    @abstractmethod
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculates evaluation scores.
        Args:
            y_true: np.ndarray: True target values.
            y_pred: np.ndarray: Predicted target values.
        Returns:
            float: Evaluation scores.
        """
        pass

class MSE(EvaluationStrategy):
    """
    Concrete class for Mean Squared Error.
    """
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray)-> float:
        """
        Calculates Mean Squared Error.
        Args:
            y_true: np.ndarray: True target values.
            y_pred: np.ndarray: Predicted target values.
        Returns:
            float: Evaluation scores.
        """
        try:
            logging.info("Calculating Mean Squared Error.")
            mse = mean_squared_error(y_true, y_pred)
            logging.info("Mean Squared Error: {}".format(mse))
            return mse
        except Exception as e:
            logging.error("Error in calculating Mean Squared Error: {}".format(e))
            return e 
        
class R2(EvaluationStrategy):
    """
    Concrete class for R2 Score.
    """
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray)-> float:
        """
        Calculates R2 Score.
        Args:
            y_true: np.ndarray: True target values.
            y_pred: np.ndarray: Predicted target values.
        Returns:
            float: Evaluation scores.
        """
        try:
            logging.info("Calculating R2 Score.")
            r2 = r2_score(y_true, y_pred)
            logging.info("R2 Score: {}".format(r2))
            return r2
        except Exception as e:
            logging.error("Error in calculating R2 Score: {}".format(e))
            return e

class MAE(EvaluationStrategy):
    """
    Concrete class for Mean Absolute Error.
    """
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray)-> float:
        """
        Calculates Mean Absolute Error.
        Args:
            y_true: np.ndarray: True target values.
            y_pred: np.ndarray: Predicted target values.
        Returns:
            float: Evaluation scores.
        """
        try:
            logging.info("Calculating Mean Absolute Error.")
            mae = mean_absolute_error(y_true, y_pred)
            logging.info("Mean Absolute Error: {}".format(mae))
            return mae
        except Exception as e:
            logging.error("Error in calculating Mean Absolute Error: {}".format(e))
            return e