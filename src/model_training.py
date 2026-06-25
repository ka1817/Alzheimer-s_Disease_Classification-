import logging
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import recall_score, classification_report, make_scorer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


class ModelTraining:

    def __init__(self):
        self.best_model = None

    def build_pipeline(self):
        """
        Scaling + Random Forest pipeline
        """
        return Pipeline([
            ("scaler", StandardScaler()),
            ("rf", RandomForestClassifier(random_state=42))
        ])

    def randomforest_hyperparameter_tuning(self, X_train, y_train):
        """
        Hyperparameter tuning using RECALL (NOT accuracy)
        """

        logging.info("Starting hyperparameter tuning (Recall optimized)...")

        pipeline = self.build_pipeline()

        # Use recall as scoring metric
        recall_scorer = make_scorer(recall_score)

        param_grid = {
            "rf__n_estimators": [100, 200],
            "rf__max_depth": [None, 10, 20],
            "rf__min_samples_split": [2, 5],
            "rf__min_samples_leaf": [1, 2]
        }

        grid_search = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            cv=5,
            scoring=recall_scorer,
            n_jobs=-1,
            verbose=1
        )

        grid_search.fit(X_train, y_train)

        self.best_model = grid_search.best_estimator_

        logging.info(f"Best parameters: {grid_search.best_params_}")
        logging.info(f"Best CV Recall: {grid_search.best_score_}")

        return self.best_model
    def gradientboosting_hyperparameter_tuning(self, X_train, y_train):

        logging.info("Starting Gradient Boosting hyperparameter tuning (Recall optimized)...")

        pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("gb", GradientBoostingClassifier(random_state=42))
    ])

        recall_scorer = make_scorer(recall_score)

        param_grid = {
        "gb__n_estimators": [100, 200, 300],
        "gb__learning_rate": [0.01, 0.05, 0.1],
        "gb__max_depth": [3, 5, 7],
        "gb__min_samples_split": [2, 5],
        "gb__min_samples_leaf": [1, 2]
    }

        grid_search = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            cv=5,
            scoring=recall_scorer,
            n_jobs=-1,
            verbose=1
    )

        grid_search.fit(X_train, y_train)

        self.best_model = grid_search.best_estimator_

        logging.info(f"Best parameters: {grid_search.best_params_}")
        logging.info(f"Best CV Recall: {grid_search.best_score_}")

        return self.best_model

    def evaluate(self, model, X_test, y_test, threshold=0.35):

        logging.info(f"Applying threshold: {threshold}")

        y_proba = model.predict_proba(X_test)[:, 1]

        y_pred = (y_proba >= threshold).astype(int)

        recall = recall_score(y_test, y_pred)

        logging.info(f"Recall: {recall}")
        logging.info("Classification Report:")
        logging.info("\n" + classification_report(y_test, y_pred))

        return recall