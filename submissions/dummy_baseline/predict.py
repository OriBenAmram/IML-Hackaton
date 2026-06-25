import joblib
import numpy as np
import pandas as pd

from base_model import BaseModel
from model import BikeDemandModel


class Model(BaseModel):
    """
    Fixed grader-facing wrapper.

    The evaluator calls:

        model = Model()
        model.load("weights.joblib")
        preds = model.predict(hidden_test_targets_df)

    Students should generally edit model.py and train.py, not this file.
    """

    def __init__(self):
        self.model = BikeDemandModel()

    def load(self, weights_path: str) -> None:
        artifacts = joblib.load(weights_path)
        self.model.load_artifacts(artifacts)

    def predict(self, test_df: pd.DataFrame) -> np.ndarray:
        return self.model.predict(test_df)