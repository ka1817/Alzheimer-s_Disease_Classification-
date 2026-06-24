import pandas as pd
import logging
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DataIngestion:

   
    def __init__(self, path: str):
        
        self.path = path

    def load_data(self) -> pd.DataFrame:
        
        logging.info(f"Attempting to load data from: {self.path}")
        
        if not os.path.exists(self.path):
            logging.error(f"File not found: {self.path}")
            raise FileNotFoundError(f"The file at {self.path} does not exist.")

        try:
            df = pd.read_csv(self.path)
            logging.info(f"Data successfully loaded. Shape: {df.shape}")
            return df
          
        except pd.errors.EmptyDataError as e:
            logging.error(f"The CSV file at {self.path} is empty.")
            raise e
            
        except Exception as e:
            logging.error(f"An unexpected error occurred while loading the data: {e}")
            raise e

if __name__ == "__main__":
    ingestion = DataIngestion("..\P685_alzheimers_disease_data.csv")
    df=ingestion.load_data()
    print(df.shape)
