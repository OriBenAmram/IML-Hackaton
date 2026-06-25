#!/usr/bin/env python3
"""
Training script for the bike-demand submission.

Run from this folder:

    cd submissions/YOUR_TEAM_NAME
    python train.py

Expected dataset:

    ../../dataset/train_set.csv

Output:

    weights.joblib

The evaluator will later load weights.joblib through predict.py.
"""

from pathlib import Path

import joblib
import pandas as pd


DATA_ROOT = Path("../../dataset")
TRAIN_CSV = DATA_ROOT / "train_set.csv"
OUTPUT_WEIGHTS = "weights.joblib"


def main() -> None:
    train = pd.read_csv(TRAIN_CSV, low_memory=False)

    # TODO: Create your training features.
    # Example:
    # X_train = create_features(train)

    # TODO: Create your training target.
    # Example:
    # y_train = train["demand"]

    # TODO: Train your model.
    # Example:
    # model.fit(X_train, y_train)

    # TODO: Save every object needed later during prediction.
    # This can include:
    #   - trained model
    #   - feature column names
    #   - scalers / encoders
    #   - lookup tables
    #   - medians / fallback values

    artifacts = {
        "model": None,
        "feature_columns": [],
    }

    joblib.dump(artifacts, OUTPUT_WEIGHTS)

    print(f"Saved {OUTPUT_WEIGHTS}")


if __name__ == "__main__":
    main()