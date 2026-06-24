import pandas as pd
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class FeatureSelection:

    def __init__(self, dataframe: pd.DataFrame):
        self.df = dataframe

        # define selected features once (clean + maintainable)
        self.selected_features = [
            "FunctionalAssessment",
            "ADL",
            "MemoryComplaints",
            "MMSE",
            "BehavioralProblems",
            "SleepQuality",
            "CholesterolHDL",
            "Diagnosis"
        ]

    def select_features(self) -> pd.DataFrame:
        """
        Returns dataframe with only selected features.
        """

        logging.info("Starting feature selection...")

        # check missing columns (important for real projects)
        missing_cols = [
            col for col in self.selected_features
            if col not in self.df.columns
        ]

        if missing_cols:
            logging.error(f"Missing columns in dataset: {missing_cols}")
            raise KeyError(f"Missing columns: {missing_cols}")

        df_new = self.df[self.selected_features].copy()

        logging.info(f"Feature selection completed. New shape: {df_new.shape}")

        return df_new


if __name__ == "__main__":

    from src.data_ingestion import DataIngestion

    ingestion = DataIngestion("P685_alzheimers_disease_data.csv")
    df = ingestion.load_data()

    feature_selector = FeatureSelection(df)
    df_new = feature_selector.select_features()

    print(df_new.head())
    print("Shape:", df_new.shape)