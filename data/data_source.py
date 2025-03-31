import pandas as pd
from utils.logging import setup_logging
from .indicators import calculate_indicators

#logger = setup_logging(__name__)

class CSVDataSource:
    def __init__(self, filename: str = "bitcoin_data_cleaned.csv"):
        self.__filename = filename
    
    def get_data(self) -> pd.DataFrame:
        #logger.info(f"Loading data from {self.__filename}")
        try:
            data = pd.read_csv(self.__filename)
            data['timestamp'] = pd.to_datetime(data['timestamp'])
            #logger.info(f"Loaded {len(data)} rows from CSV")
        except Exception as e:
            raise ValueError(f"Error loading CSV: {str(e)}")
        data = data.dropna()
        return calculate_indicators(data)