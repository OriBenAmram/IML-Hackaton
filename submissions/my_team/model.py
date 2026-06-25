import numpy as np
import pandas as pd

# ── Feature column definitions (single source of truth) ──────────────────────
# train.py imports these to stay in sync at training time.

CAT_FEAT_COLS = ["city", "start_station_id"]

NUMERIC_FEAT_COLS = [
    # Cyclical time: 23:00 and 00:00 are adjacent; Sunday and Monday are adjacent
    "hour_sin", "hour_cos", "weekday_sin", "weekday_cos",
    # Calendar flags
    "is_weekend", "is_holiday", "is_working_day",
    # Station location
    "start_lat", "start_lng",
    # Weather
    "temperature_2m", "apparent_temperature", "relative_humidity_2m",
    "precipitation", "rain", "snowfall", "cloud_cover", "wind_speed_10m",
    # Station infrastructure / POIs
    "bike_lane_length_500m", "park_area_500m",
    "university_count_1000m", "office_poi_count_1000m", "retail_poi_count_1000m",
    "restaurant_cafe_count_500m", "transit_stop_count_500m",
    "distance_to_nearest_rail_station", "distance_to_city_center",
    # Hierarchical statistical baseline prediction (the key generalisation feature)
    "stat_baseline_pred",
]

# CatBoost pool column order: categoricals first, then numerics
ALL_FEAT_COLS = CAT_FEAT_COLS + NUMERIC_FEAT_COLS

# ── Stat-baseline fallback hierarchy ─────────────────────────────────────────
# Most specific → least specific. L7 is the global mean scalar (handled separately).
STAT_LEVELS = [
    ("L1", ["city", "start_station_id", "hour", "weekday"]),
    ("L2", ["city", "start_station_id", "hour"]),
    ("L3", ["city", "start_station_id"]),
    ("L4", ["city", "hour", "weekday"]),
    ("L5", ["city", "hour"]),
    ("L6", ["hour", "weekday"]),
]


# ── Shared helpers (train.py imports these) ───────────────────────────────────

def add_cyclical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds sin/cos encodings for hour (period=24) and weekday (period=7).
    Returns a copy with four new columns; original df is not mutated.
    Requires columns: hour (0–23), weekday (0–6).
    """
    df = df.copy()
    df["hour_sin"]    = np.sin(2 * np.pi * df["hour"]    / 24)
    df["hour_cos"]    = np.cos(2 * np.pi * df["hour"]    / 24)
    df["weekday_sin"] = np.sin(2 * np.pi * df["weekday"] / 7)
    df["weekday_cos"] = np.cos(2 * np.pi * df["weekday"] / 7)
    return df


def apply_stat_baseline(df: pd.DataFrame, stat_tables: dict) -> np.ndarray:
    """
    Hierarchical mean-demand lookup with automatic fallback.

    Requires columns: city, start_station_id, hour (int), weekday (int).
    stat_tables keys: L1–L6 (plain dicts with tuple keys), L7 (float scalar).

    For unseen (city, station) combinations the chain falls through to
    global hour+weekday means (L6) or the global mean (L7), so predictions
    degrade gracefully on new cities/stations.
    """
    # Work with a clean, normalised copy of the key columns
    work = pd.DataFrame({
        "city":             df["city"].astype(str).values,
        "start_station_id": df["start_station_id"].astype(str).values,
        "hour":             df["hour"].astype(int).values,
        "weekday":          df["weekday"].astype(int).values,
    })

    pred = np.full(len(work), np.nan, dtype=np.float64)

    for level_key, cols in STAT_LEVELS:
        remaining = np.isnan(pred)
        if not remaining.any():
            break
        lookup = stat_tables[level_key]  # plain dict: tuple → float
        keys = list(zip(*[work[c].tolist() for c in cols]))
        vals = np.array([lookup.get(k, np.nan) for k in keys], dtype=np.float64)
        pred[remaining] = vals[remaining]

    still_nan = np.isnan(pred)
    if still_nan.any():
        pred[still_nan] = float(stat_tables["L7"])

    return pred


# ── Model class ───────────────────────────────────────────────────────────────

class BikeDemandModel:
    """
    Blended model: alpha * CatBoost + (1 - alpha) * hierarchical stat baseline.
    """

    def __init__(self):
        self.model       = None
        self.stat_tables = None
        self.alpha       = 0.5
        self.station_meta = None

    def load_artifacts(self, artifacts: dict) -> None:
        self.model        = artifacts["model"]
        self.stat_tables  = artifacts["stat_tables"]
        self.alpha        = float(artifacts["alpha"])
        self.station_meta = artifacts.get("station_meta")

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _ensure_time_cols(self, df: pd.DataFrame) -> pd.DataFrame:
        """Derive hour and weekday from a timestamp column if not already present."""
        if "hour" not in df.columns or "weekday" not in df.columns:
            ts_col = "target_hour_start" if "target_hour_start" in df.columns else "hour_ts"
            ts = pd.to_datetime(df[ts_col], errors="coerce")
            df = df.copy()
            df["hour"]    = ts.dt.hour
            df["weekday"] = ts.dt.weekday
        return df

    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform a raw station-hour DataFrame into one containing all ALL_FEAT_COLS.
        Does not mutate the input.
        """
        df = self._ensure_time_cols(df)
        df = df.copy()

        # Rename raw calendar columns to our internal names
        df.rename(columns={
            "weekend":     "is_weekend",
            "holiday":     "is_holiday",
            "working_day": "is_working_day",
        }, inplace=True)

        # Cyclical time encodings
        df = add_cyclical_features(df)

        # Normalise categorical IDs to strings (CatBoost expects consistent types)
        df["city"]             = df["city"].astype(str)
        df["start_station_id"] = df["start_station_id"].astype(str)

        # Statistical baseline as a numeric feature
        df["stat_baseline_pred"] = apply_stat_baseline(df, self.stat_tables)

        # Ensure all numeric feature columns are present (fill NaN if missing)
        for col in NUMERIC_FEAT_COLS:
            if col not in df.columns:
                df[col] = np.nan

        return df

    # ── Public interface ─────────────────────────────────────────────────────

    def predict(self, test_df: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_artifacts() first.")

        df = self._engineer_features(test_df)

        stat_pred = df["stat_baseline_pred"].values

        from catboost import Pool
        pool = Pool(data=df[ALL_FEAT_COLS], cat_features=CAT_FEAT_COLS)
        tree_pred = self.model.predict(pool)

        blended = self.alpha * tree_pred + (1.0 - self.alpha) * stat_pred
        return np.clip(blended, 0.0, None)
