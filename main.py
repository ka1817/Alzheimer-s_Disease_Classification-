# main.py

import logging
import os
import joblib

from src.data_ingestion import DataIngestion
from src.feature_selection import FeatureSelection
from src.data_splitting import DataSplitting
from src.model_training import ModelTraining


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


def main():

    logging.info("Step 1: Data Ingestion")

    ingestion = DataIngestion("P685_alzheimers_disease_data.csv")
    df = ingestion.load_data()

    logging.info("Step 2: Feature Selection")

    feature_selector = FeatureSelection(df)
    df_selected = feature_selector.select_features()

    logging.info("Step 3: Data Splitting")

    splitter = DataSplitting(
        df_selected,
        target_column="Diagnosis"
    )

    X_train, X_test, y_train, y_test = splitter.split_data()

    logging.info("Step 4: Training Models")

    trainer = ModelTraining()

    rf_model = trainer.randomforest_hyperparameter_tuning(
        X_train,
        y_train
    )

    gb_model = trainer.gradientboosting_hyperparameter_tuning(
        X_train,
        y_train
    )

    logging.info("Step 5: Evaluating Models")

    rf_recall = trainer.evaluate(
        rf_model,
        X_test,
        y_test,
        threshold=0.35
    )

    gb_recall = trainer.evaluate(
        gb_model,
        X_test,
        y_test,
        threshold=0.35
    )

    logging.info(f"Random Forest Recall: {rf_recall:.4f}")
    logging.info(f"Gradient Boosting Recall: {gb_recall:.4f}")

    logging.info("Step 6: Saving Models")

    os.makedirs("models", exist_ok=True)

    joblib.dump(
        rf_model,
        "models/RandomForest_Alzheimers_model.pkl"
    )

    joblib.dump(
        gb_model,
        "models/GradientBoosting_Alzheimers_model.pkl"
    )

    logging.info("Both models saved successfully.")

    
if __name__ == "__main__":
    main()