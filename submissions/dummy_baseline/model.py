import numpy as np
import pandas as pd


class BikeDemandModel:
    """
    Actual dummy model logic.

    This model ignores the input features and always predicts the same constant.
    """

    def __init__(self):
        self.constant_prediction = None

    def load_artifacts(self, artifacts: dict) -> None:
        self.constant_prediction = float(artifacts["constant_prediction"])

    def predict(self, test_df: pd.DataFrame) -> np.ndarray:
        if self.constant_prediction is None:
            raise RuntimeError("Model is not loaded. Call load_artifacts() first.")

        return np.full(
            shape=len(test_df),
            fill_value=self.constant_prediction,
            dtype=float,
        )