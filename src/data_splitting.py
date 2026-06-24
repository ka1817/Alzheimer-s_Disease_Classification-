import pandas as pd
import logging
from sklearn.model_selection import train_test_split

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

class DataSplitting:

    def __init__(self, dataframe: pd.DataFrame, target_column: str = "Diagnosis"):
        self.df = dataframe
        self.target_column = target_column

    def split_data(self, test_size: float = 0.2, random_state: int = 42):
        """
        Splits dataset into train and test sets.
        
        Returns:
            X_train, X_test, y_train, y_test
        """

        logging.info("Starting data splitting...")

        if self.target_column not in self.df.columns:
            logging.error(f"Target column '{self.target_column}' not found in dataframe.")
            raise KeyError(f"Missing target column: {self.target_column}")

        X = self.df.drop(columns=[self.target_column])
        y = self.df[self.target_column]

        logging.info(f"Feature matrix shape: {X.shape}, Target shape: {y.shape}")

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state
        )

        logging.info(f"Data split completed:")
        logging.info(f"X_train: {X_train.shape}, X_test: {X_test.shape}")
        logging.info(f"y_train: {y_train.shape}, y_test: {y_test.shape}")

        return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    from src.data_ingestion import DataIngestion
    from src.feature_selection import FeatureSelection

    ingestion = DataIngestion(r"P685_alzheimers_disease_data.csv")
    df = ingestion.load_data()

    fs = FeatureSelection(df)
    df_selected = fs.select_features()

    splitter = DataSplitting(df_selected)
    X_train, X_test, y_train, y_test = splitter.split_data()

    print(X_train.shape, X_test.shape)