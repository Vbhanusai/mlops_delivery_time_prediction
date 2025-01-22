import logging
import pandas as pd
from zenml import step
from src.cleaning_data import DataCleaning, DataDivideStrategy, DataPreProcessStrategy

from typing import Tuple
from typing_extensions import Annotated
@step
def clean_data(df: pd.DataFrame) -> Tuple[
    Annotated[pd.DataFrame, "X_train"],
    Annotated[pd.DataFrame, "X_test"],
    Annotated[pd.Series, "y_train"],
    Annotated[pd.Series , "y_test"]
]:
    """
    Cleans data and divides it into training and testing sets.
    Args:
        df: pd.DataFrame: Input data.
    Returns:
        X_train: pd.DataFrame: Training features.
        X_test: pd.DataFrame: Testing features.
        y_train: pd.Series: Training target.
        y_test: pd.Series: Testing target.
    """
    try:
        process_strategy = DataPreProcessStrategy()
        data_cleaning = DataCleaning(df, process_strategy)
        processed_data = data_cleaning.handle_data()
        
        divide_strategy = DataDivideStrategy()
        data_cleaning = DataCleaning(processed_data, divide_strategy)
        X_train, X_test, y_train,y_test = data_cleaning.handle_data()
        logging.info("Data cleaning completed.")
        return X_train, X_test, y_train, y_test
    except Exception as e:
        logging.error("Error in cleaning data: {}".format(e))
        return e