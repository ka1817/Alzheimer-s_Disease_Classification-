# main.py
import logging
import joblib
import os
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
    ingestion = DataIngestion(r"P685_alzheimers_disease_data.csv")
    df = ingestion.load_data()

    logging.info("Step 2: Feature Selection")
    feature_selector = FeatureSelection(df)
    df_selected = feature_selector.select_features()

    logging.info("Step 3: Data Splitting")
    splitter = DataSplitting(df_selected, target_column="Diagnosis")
    X_train, X_test, y_train, y_test = splitter.split_data()

    logging.info("Step 4: Model Training")
    trainer = ModelTraining()

    best_model = trainer.hyperparameter_tuning(X_train, y_train)

    logging.info("Step 5: Evaluation")
    recall = trainer.evaluate(best_model, X_test, y_test, threshold=0.35)

    logging.info(f"Final Test Recall: {recall}")

    print("\nPipeline completed successfully!")
    print(f"Final Recall: {recall}")

    

    logging.info("Step 6: Saving Model")

    os.makedirs("models", exist_ok=True)

    joblib.dump(best_model, "models/alzheimers_model.pkl")

    logging.info("Model saved at models/alzheimers_model.pkl")

    print("\nPipeline completed successfully!")
    print(f"Final Recall: {recall}")


if __name__ == "__main__":
    main()