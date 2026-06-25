#!/usr/bin/env python3
"""
Dummy training file for the bike-demand submission format.

This is intentionally a real dummy:
it does not train a model.

It only saves a constant prediction value into weights.joblib.

Run from this folder:

    cd submissions/dummy_constant_100
    python train.py

Output:

    weights.joblib
"""

from pathlib import Path

import joblib
import pandas as pd


DATA_ROOT = Path("../../dataset")
TRAIN_CSV = DATA_ROOT / "train_set.csv"
OUTPUT_WEIGHTS = "weights.joblib"


def main() -> None:
    # Read the train file only to verify that the expected dataset exists.
    # The dummy model does not use it.
    train = pd.read_csv(TRAIN_CSV, low_memory=False)

    artifacts = {
        "constant_prediction": 100.0,
    }

    joblib.dump(artifacts, OUTPUT_WEIGHTS)

    print(f"Read {len(train):,} training rows from {TRAIN_CSV}")
    print(f"Saved {OUTPUT_WEIGHTS}")
    print("This dummy model always predicts 100.")


if __name__ == "__main__":
    main()