import logging
import pandas as pd
from src.cleaning_data import DataCleaning, DataPreProcessStrategy

def get_data_for_test():
    try:
        data = pd.read_csv("data/Food_Delivery_Times.csv")
        data = data.sample(n=100)
        preprocess_strategy = DataPreProcessStrategy()
        data_cleaning = DataCleaning(data, preprocess_strategy)
        data = data_cleaning.handle_data()
        data.drop(['Delivery_Time_min'], axis=1, inplace=True)
        result = data.to_json(orient="split")
        return result
    except Exception as e:
        logging.error(f"Error in get_data_for_test: {str(e)}")
        return e
    
