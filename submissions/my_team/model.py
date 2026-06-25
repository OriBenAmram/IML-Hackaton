import numpy as np
import pandas as pd


class BikeDemandModel:
    """
    Put your actual model logic here.

    This file should contain:
        - feature creation used during prediction
        - model-specific preprocessing
        - prediction logic

    Do NOT load weights.joblib here.
    The weights are loaded in predict.py and passed into this class.
    """

    def __init__(self):
        self.artifacts = None

    def load_artifacts(self, artifacts: dict) -> None:
        """
        Store all objects created by train.py.

        Examples:
            artifacts["model"]
            artifacts["feature_columns"]
            artifacts["scaler"]
        """
        self.artifacts = artifacts

    def predict(self, test_df: pd.DataFrame) -> np.ndarray:
        """
        Predict bike demand for each row in test_df.

        Parameters
        ----------
        test_df:
            Hidden station-hour test features provided by the evaluator.
            It does NOT contain the demand column.

        Returns
        -------
        np.ndarray:
            One numeric prediction per row in test_df.
        """
        if self.artifacts is None:
            raise RuntimeError("Model is not loaded. Call load_artifacts() first.")

        # TODO: Build the same features used during training.
        # Example:
        # X = create_features(test_df)

        # TODO: Load/use your trained model or lookup tables from self.artifacts.
        # Example:
        # model = self.artifacts["model"]
        # preds = model.predict(X)

        # TODO: Replace this placeholder with your actual predictions.
        preds = np.zeros(len(test_df), dtype=float)

        # Bike demand cannot be negative.
        return np.maximum(0.0, preds)