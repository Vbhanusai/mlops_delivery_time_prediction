import logging
from abc import ABC, abstractmethod
from typing import Union

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

class DataStrategy(ABC):
    """
    Abstract class for data strategies.
    """
    @abstractmethod
    def handle_data(self, data: pd.DataFrame) -> Union[pd.DataFrame,pd.Series]:
        pass

class DataPreProcessStrategy(DataStrategy):
    """
    Concrete class for preprocessing data.
    """
    def handle_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocesses data.
        Args:
            data: pd.DataFrame: Input data.
        Returns:
            pd.DataFrame: Preprocessed data.
        """
        # Drop Unnecessary Columns
        data.drop(columns=['Order_ID'], axis=1, inplace=True)   
             
        # Drop duplicate rows
        data = data.drop_duplicates()

        numeric_features = data.select_dtypes(exclude=['object']).columns
        categorical_features = data.select_dtypes(include=['object']).columns

        # Impute missing values
        numeric_imputer = SimpleImputer(strategy='median')
        categorical_imputer = SimpleImputer(strategy='most_frequent')

        data[numeric_features] = numeric_imputer.fit_transform(data[numeric_features])
        data[categorical_features] = categorical_imputer.fit_transform(data[categorical_features])

        data = pd.get_dummies(
            data,
            columns=categorical_features,
            drop_first=True)
        return data

class DataDivideStrategy(DataStrategy):
    """
    Concrete class for dividing data.
    """
    def handle_data(self, data: pd.DataFrame) -> Union[pd.DataFrame,pd.Series]:
        """
        Divides data into features and target.
        Args:
            data: pd.DataFrame: Input data.
        Returns:
            Union[pd.DataFrame,pd.Series]: Features and target.
        """
        try:
            # Divide data into features and target
            X = data.drop( columns=['Delivery_Time_min'], axis=1, inplace=False)
            y = data['Delivery_Time_min']


            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            return X_train, X_test, y_train, y_test
        
        except KeyError as e:
            logging.error("Error in dividing data: {}".format(e))
            return e

class DataCleaning:
    """
    Class for data cleaning which processes the data and divides into training and testing sets.
    """
    def __init__(self, data: pd.DataFrame, strategy: DataStrategy):
        self.data = data
        self.strategy = strategy

    def handle_data(self) -> Union[pd.DataFrame,pd.Series]:
        """
        Cleans data.
        Returns:
            Union[pd.DataFrame,pd.Series]: Cleaned data.
        """
        try:
            return self.strategy.handle_data(self.data)
        except Exception as e:
            logging.error("Error in cleaning data: {}".format(e))
            return e