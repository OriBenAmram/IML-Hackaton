#!/usr/bin/env python3
"""
Training script for the bike-demand submission.

Run from this folder:
    cd submissions/my_team
    python train.py

Reads:  ../../dataset/local_train_set.csv
Writes: weights.joblib

Bump N_ROWS_CAP to scale up the tree training set:
    100_000  →  fast pipeline check (~10 s fit)
    300_000  →  better generalisation (~1 min)
    600_000  →  strong run (~3 min)
"""

from pathlib import Path
import sys
import numpy as np
import pandas as pd
import joblib
from catboost import CatBoostRegressor, Pool

# Import shared constants and helpers from model.py (same directory)
sys.path.insert(0, str(Path(__file__).parent))
from model import (
    add_cyclical_features,
    apply_stat_baseline,
    STAT_LEVELS,
    CAT_FEAT_COLS,
    NUMERIC_FEAT_COLS,
    ALL_FEAT_COLS,
)

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_ROOT   = Path("../../dataset")
TRAIN_CSV   = DATA_ROOT / "local_train_set.csv"
VAL_TARGETS = DATA_ROOT / "public_validation_targets.csv"
VAL_LABELS  = DATA_ROOT / "private_labels.csv"
OUTPUT      = Path("weights.joblib")

# ── Cap on tree training rows ─────────────────────────────────────────────────
N_ROWS_CAP = 900_000

# ── Column groups ─────────────────────────────────────────────────────────────
STATION_META_COLS = [
    "start_station_id", "city",
    "start_lat", "start_lng",
    "bike_lane_length_500m", "park_area_500m",
    "university_count_1000m", "office_poi_count_1000m", "retail_poi_count_1000m",
    "restaurant_cafe_count_500m", "transit_stop_count_500m",
    "distance_to_nearest_rail_station", "distance_to_city_center",
]
WEATHER_COLS = [
    "city", "hour_ts",
    "temperature_2m", "apparent_temperature", "relative_humidity_2m",
    "precipitation", "rain", "snowfall", "cloud_cover", "wind_speed_10m",
]
CALENDAR_COLS = ["hour_ts", "weekday", "weekend", "holiday", "working_day"]


# ── Data loading ──────────────────────────────────────────────────────────────

def load_rides(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    df["hour_ts"] = pd.to_datetime(df["hour_ts"], errors="coerce")
    df = df.dropna(subset=["hour_ts"]).reset_index(drop=True)
    df["hour"]    = df["hour_ts"].dt.hour
    df["weekday"] = df["hour_ts"].dt.weekday
    df["start_station_id"] = df["start_station_id"].astype(str)
    return df


# ── Station-hour demand grid ──────────────────────────────────────────────────

def build_demand_grid(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cross each unique station with every city-hour observed in training.
    Unobserved (station, hour_ts) pairs get demand = 0.
    This mirrors the active-window grid used by the evaluator, keeping the
    training distribution consistent with what the model is scored on.
    """
    records = []
    for city, grp in df.groupby("city"):
        stations   = grp["start_station_id"].unique()
        city_hours = grp["hour_ts"].unique()

        idx = pd.MultiIndex.from_product(
            [stations, city_hours],
            names=["start_station_id", "hour_ts"],
        )
        full = pd.DataFrame(index=idx).reset_index()
        full["city"] = city

        counts = (
            grp.groupby(["start_station_id", "hour_ts"])
               .size()
               .reset_index(name="demand")
        )
        full = full.merge(counts, on=["start_station_id", "hour_ts"], how="left")
        full["demand"] = full["demand"].fillna(0).astype(np.int32)
        records.append(full)

    return pd.concat(records, ignore_index=True)


# ── Statistical baseline ──────────────────────────────────────────────────────

def build_stat_tables(demand_df: pd.DataFrame) -> dict:
    """
    Compute hierarchical mean-demand lookup tables from the full demand grid.
    L1–L6 are stored as plain dicts (tuple keys → float) for O(1) inference.
    L7 is the global mean scalar used as the final fallback.
    """
    tables = {}
    for level_key, cols in STAT_LEVELS:
        tables[level_key] = (
            demand_df.groupby(cols)["demand"].mean().to_dict()
        )
    tables["L7"] = float(demand_df["demand"].mean())
    return tables


# ── Feature engineering ───────────────────────────────────────────────────────

def prepare_features(demand_df: pd.DataFrame, df_rides: pd.DataFrame) -> pd.DataFrame:
    """
    Join station metadata, weather, and calendar features onto the demand grid,
    then add cyclical hour/weekday encodings.
    """
    # Station metadata: constant per (start_station_id, city)
    avail_meta = [c for c in STATION_META_COLS if c in df_rides.columns]
    station_meta = (
        df_rides[avail_meta]
        .drop_duplicates(["start_station_id", "city"])
        .reset_index(drop=True)
    )
    demand_df = demand_df.merge(station_meta, on=["start_station_id", "city"], how="left")

    # Weather: per (city, hour_ts)
    avail_wx = [c for c in WEATHER_COLS if c in df_rides.columns]
    weather = (
        df_rides[avail_wx]
        .drop_duplicates(["city", "hour_ts"])
        .reset_index(drop=True)
    )
    demand_df = demand_df.merge(weather, on=["city", "hour_ts"], how="left")

    # Calendar: per hour_ts (weekday, weekend, holiday, working_day)
    avail_cal = [c for c in CALENDAR_COLS if c in df_rides.columns]
    calendar = (
        df_rides[avail_cal]
        .drop_duplicates("hour_ts")
        .reset_index(drop=True)
    )
    demand_df = demand_df.merge(calendar, on="hour_ts", how="left")

    # Derive hour + weekday from hour_ts (authoritative; overwrites the joined copy)
    demand_df["hour"]    = demand_df["hour_ts"].dt.hour
    demand_df["weekday"] = demand_df["hour_ts"].dt.weekday

    # Rename raw calendar column names to our internal convention
    demand_df.rename(columns={
        "weekend":     "is_weekend",
        "holiday":     "is_holiday",
        "working_day": "is_working_day",
    }, inplace=True)

    # Cyclical time encodings
    demand_df = add_cyclical_features(demand_df)

    return demand_df


# ── Subsampling ───────────────────────────────────────────────────────────────

def subsample_for_tree(demand_df: pd.DataFrame, n_cap: int) -> pd.DataFrame:
    """
    Take the most recent n_cap rows by hour_ts.
    Recent demand is closest to the validation period and preserves the natural
    zero/nonzero ratio rather than forcing an artificial 50/50 balance.
    """
    return (
        demand_df.sort_values("hour_ts")
        .tail(n_cap)
        .reset_index(drop=True)
    )


# ── Validation-set feature engineering (mirrors _engineer_features in model.py) ─

def engineer_val_features(
    val: pd.DataFrame,
    stat_tables: dict,
) -> pd.DataFrame:
    """
    Apply the same transformations as BikeDemandModel._engineer_features
    so predictions on the validation set are comparable.
    """
    val = val.copy()

    # Derive time columns if missing
    if "hour" not in val.columns or "weekday" not in val.columns:
        ts_col = "hour_ts" if "hour_ts" in val.columns else "target_hour_start"
        ts = pd.to_datetime(val[ts_col], errors="coerce")
        val["hour"]    = ts.dt.hour
        val["weekday"] = ts.dt.weekday

    val.rename(columns={
        "weekend":     "is_weekend",
        "holiday":     "is_holiday",
        "working_day": "is_working_day",
    }, inplace=True)

    val["start_station_id"] = val["start_station_id"].astype(str)
    val["city"]             = val["city"].astype(str)

    val = add_cyclical_features(val)
    val["stat_baseline_pred"] = apply_stat_baseline(val, stat_tables)

    # Ensure all numeric columns exist
    for col in NUMERIC_FEAT_COLS:
        if col not in val.columns:
            val[col] = np.nan

    return val


# ── Alpha tuning ──────────────────────────────────────────────────────────────

def tune_alpha(
    tree_preds: np.ndarray,
    stat_preds: np.ndarray,
    y_true: np.ndarray,
    n_steps: int = 21,
) -> tuple:
    best_alpha, best_mae = 0.0, np.inf
    for alpha in np.linspace(0.0, 1.0, n_steps):
        blended = np.clip(alpha * tree_preds + (1.0 - alpha) * stat_preds, 0.0, None)
        mae = float(np.mean(np.abs(blended - y_true)))
        if mae < best_mae:
            best_mae, best_alpha = mae, float(alpha)
    return best_alpha, best_mae


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 60)
    print(f"N_ROWS_CAP = {N_ROWS_CAP:,}  (edit to scale up)")
    print("=" * 60)

    # ── 1. Load ride data ────────────────────────────────────────────────────
    print("\n[1] Loading ride data...")
    df_rides = load_rides(TRAIN_CSV)
    print(f"    {len(df_rides):,} rides | cities: {sorted(df_rides['city'].unique())}")

    # Station metadata saved separately as an artifact
    avail_meta = [c for c in STATION_META_COLS if c in df_rides.columns]
    station_meta = (
        df_rides[avail_meta]
        .drop_duplicates(["start_station_id", "city"])
        .reset_index(drop=True)
    )

    # ── 2. Build station-hour demand grid (with zero-demand rows) ────────────
    print("\n[2] Building station-hour demand grid...")
    demand_df = build_demand_grid(df_rides)
    n_zero    = (demand_df["demand"] == 0).sum()
    n_nonzero = (demand_df["demand"] >  0).sum()
    print(f"    Grid: {len(demand_df):,} rows  |  nonzero: {n_nonzero:,}  |  zeros: {n_zero:,}")

    # ── 3. Derive time keys before building stat tables ──────────────────────
    demand_df["hour"]    = demand_df["hour_ts"].dt.hour
    demand_df["weekday"] = demand_df["hour_ts"].dt.weekday

    # ── 4. Build statistical baseline (uses the full grid, including zeros) ──
    print("\n[3] Building statistical baseline tables...")
    stat_tables = build_stat_tables(demand_df)
    print(f"    L7 global mean = {stat_tables['L7']:.4f}")
    for level_key, cols in STAT_LEVELS:
        print(f"    {level_key} ({'+'.join(cols)}): {len(stat_tables[level_key]):,} entries")

    # ── 5. Join features; add cyclical encodings ─────────────────────────────
    print("\n[4] Joining features onto demand grid...")
    demand_df = prepare_features(demand_df, df_rides)

    # ── 6. Add stat_baseline_pred as a feature ───────────────────────────────
    print("\n[5] Computing stat_baseline_pred feature...")
    demand_df["stat_baseline_pred"] = apply_stat_baseline(demand_df, stat_tables)

    # ── 7. Subsample for tree training ───────────────────────────────────────
    print(f"\n[6] Subsampling for tree (most recent {N_ROWS_CAP:,} rows by hour_ts)...")
    df_tree = subsample_for_tree(demand_df, N_ROWS_CAP)
    tz = (df_tree["demand"] == 0).sum()
    tnz = (df_tree["demand"] > 0).sum()
    print(f"    Tree set: {len(df_tree):,} rows  |  nonzero: {tnz:,}  |  zeros: {tz:,}")

    # ── 8. Train CatBoostRegressor ───────────────────────────────────────────
    print("\n[7] Training CatBoostRegressor...")
    train_pool = Pool(
        data=df_tree[ALL_FEAT_COLS],
        label=df_tree["demand"].values.astype(np.float32),
        cat_features=CAT_FEAT_COLS,
    )
    model = CatBoostRegressor(
        iterations=500,
        learning_rate=0.05,
        depth=6,
        l2_leaf_reg=10,
        loss_function="MAE",
        eval_metric="MAE",
        random_seed=42,
        verbose=100,
    )
    model.fit(train_pool)

    # ── 9. Tune blend weight alpha on the local validation set ───────────────────
    print("\n[8] Tuning blend weight alpha on local validation set...")
    val_targets = pd.read_csv(VAL_TARGETS, low_memory=False)
    val_labels  = pd.read_csv(VAL_LABELS,  low_memory=False)
    val = val_targets.merge(val_labels[["id", "demand"]], on="id", how="left")
    y_val = val["demand"].values.astype(float)

    val_feats = engineer_val_features(val, stat_tables)

    val_pool        = Pool(data=val_feats[ALL_FEAT_COLS], cat_features=CAT_FEAT_COLS)
    tree_preds_val  = model.predict(val_pool)
    stat_preds_val  = val_feats["stat_baseline_pred"].values

    mae_stat = float(np.mean(np.abs(np.clip(stat_preds_val, 0, None) - y_val)))
    mae_tree = float(np.mean(np.abs(np.clip(tree_preds_val, 0, None) - y_val)))
    print(f"    alpha=0.00  (stat baseline only):  MAE = {mae_stat:.4f}")
    print(f"    alpha=1.00  (tree only):            MAE = {mae_tree:.4f}")

    best_alpha, best_mae = tune_alpha(tree_preds_val, stat_preds_val, y_val)
    print(f"    alpha={best_alpha:.2f}  (best blend):           MAE = {best_mae:.4f}")

    # ── 10. Save artifacts ───────────────────────────────────────────────────
    artifacts = {
        "model":        model,
        "stat_tables":  stat_tables,
        "alpha":        best_alpha,
        "station_meta": station_meta,
    }
    joblib.dump(artifacts, OUTPUT, compress=3)
    print(f"\nSaved {OUTPUT}  (alpha={best_alpha:.2f})")


if __name__ == "__main__":
    main()
