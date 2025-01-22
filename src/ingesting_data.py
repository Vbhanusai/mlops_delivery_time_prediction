import logging
import pandas as pd

class IngestData:
    """
    Class to ingest data from a given path
    """
    def __init__(self,data_path:str):
        """
        Args:
            data_path: Path to the data file
        """
        self.data_path = data_path
    
    def get_data(self) -> pd.DataFrame:
        """
        Ingest data from the given path"""
        logging.info(f"Running Ingest Data Step with data path: {self.data_path}")
        return pd.read_csv(self.data_path)
