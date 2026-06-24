# main.py

import logging

from src.data_ingestion import DataIngestion
from src.feature_selection import FeatureSelection
from src.data_splitting import DataSplitting
from src.model_training import ModelTraining

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


def main():

    # -----------------------
    # 1. Data Ingestion
    # -----------------------
    logging.info("Step 1: Data Ingestion")
    ingestion = DataIngestion(r"P685_alzheimers_disease_data.csv")
    df = ingestion.load_data()

    # -----------------------
    # 2. Feature Selection
    # -----------------------
    logging.info("Step 2: Feature Selection")
    feature_selector = FeatureSelection(df)
    df_selected = feature_selector.select_features()

    # -----------------------
    # 3. Train-Test Split
    # -----------------------
    logging.info("Step 3: Data Splitting")
    splitter = DataSplitting(df_selected, target_column="Diagnosis")
    X_train, X_test, y_train, y_test = splitter.split_data()

    # -----------------------
    # 4. Model Training
    # -----------------------
    logging.info("Step 4: Model Training")
    trainer = ModelTraining()

    best_model = trainer.hyperparameter_tuning(X_train, y_train)

    # -----------------------
    # 5. Evaluation
    # -----------------------
    logging.info("Step 5: Evaluation")
    recall = trainer.evaluate(best_model, X_test, y_test, threshold=0.35)

    logging.info(f"Final Test Recall: {recall}")

    print("\nPipeline completed successfully!")
    print(f"Final Recall: {recall}")


if __name__ == "__main__":
    main()