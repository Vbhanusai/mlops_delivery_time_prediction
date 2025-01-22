import logging

from zenml import step
import pandas as pd

from src.ingesting_data import IngestData

@step
def ingest_data(data_path:str) -> pd.DataFrame:
    """
    Ingest data from a given path
    Args:
        data_path: Path to the data file
    Returns:
        pd.DataFrame: Dataframe containing the data
    """
    try:
        ingest_data=IngestData(data_path)
        df = ingest_data.get_data()
        return df
    except Exception as e:
        logging.error(f"Error in Ingest Data Step: {e}")
        return e
    