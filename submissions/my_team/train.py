#!/usr/bin/env python3
"""
Training script for bike-demand prediction model.

Run from this folder:
    cd submissions/my_team
    python train.py

Expected dataset:
    ../../dataset/train_set.csv

Output:
    weights.joblib
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb


DATA_ROOT = Path("../../dataset")
TRAIN_CSV = DATA_ROOT / "train_set.csv"
OUTPUT_WEIGHTS = "weights.joblib"


def normalize_station_id(s: pd.Series) -> pd.Series:
    raw = s.astype(str).str.strip()
    numeric = pd.to_numeric(raw, errors="coerce")
    is_int = numeric.notna() & np.isfinite(numeric) & (numeric % 1 == 0)
    out = raw.copy()
    out.loc[is_int] = numeric.loc[is_int].astype(int).astype(str)
    return out


def aggregate_demand(rides: pd.DataFrame) -> pd.DataFrame:
    """Aggregate ride-level data to station-hour demand counts."""
    rides = rides.copy()
    rides["_ts"] = pd.to_datetime(rides["hour_ts"], errors="coerce")
    rides["_station"] = normalize_station_id(rides["start_station_id"])
    rides["_city"] = rides["city"].astype(str)

    # Count rides per (city, station, hour)
    demand = (
        rides.groupby(["_city", "_station", "_ts"], dropna=False)
        .size()
        .reset_index(name="demand")
    )

    # Get one row of features per (city, station, hour) via first non-null
    weather_cols = [
        "temperature_2m", "relative_humidity_2m", "apparent_temperature",
        "precipitation", "rain", "snowfall", "cloud_cover", "wind_speed_10m",
    ]
    station_cols = [
        "start_lat", "start_lng", "bike_lane_length_500m", "park_area_500m",
        "university_count_1000m", "office_poi_count_1000m", "retail_poi_count_1000m",
        "restaurant_cafe_count_500m", "transit_stop_count_500m",
        "distance_to_nearest_rail_station", "distance_to_city_center",
    ]
    calendar_cols = ["weekday", "weekend", "holiday", "holiday_name", "working_day"]

    keep = ["_city", "_station", "_ts"] + [
        c for c in weather_cols + station_cols + calendar_cols if c in rides.columns
    ]

    meta = rides[keep].groupby(["_city", "_station", "_ts"], dropna=False).first().reset_index()

    demand = demand.merge(meta, on=["_city", "_station", "_ts"], how="left")
    demand.rename(columns={"_city": "city", "_station": "station_key", "_ts": "hour_ts"}, inplace=True)
    return demand


def add_zero_demand_rows(demand: pd.DataFrame) -> pd.DataFrame:
    """Add zero-demand rows for station-hours where no rides occurred."""
    grids = []
    for city, city_df in demand.groupby("city"):
        stations = city_df["station_key"].unique()
        hours = pd.date_range(city_df["hour_ts"].min(), city_df["hour_ts"].max(), freq="h")
        # Filter to 6:00-22:00 (the evaluated range)
        hours = hours[hours.hour.isin(range(6, 23))]

        grid = pd.MultiIndex.from_product(
            [[city], stations, hours],
            names=["city", "station_key", "hour_ts"]
        ).to_frame(index=False)
        grids.append(grid)

    full_grid = pd.concat(grids, ignore_index=True)

    # Merge with existing demand
    merged = full_grid.merge(demand, on=["city", "station_key", "hour_ts"], how="left")
    merged["demand"] = merged["demand"].fillna(0).astype(int)

    # Forward fill station metadata for zero-demand rows
    station_meta_cols = [
        "start_lat", "start_lng", "bike_lane_length_500m", "park_area_500m",
        "university_count_1000m", "office_poi_count_1000m", "retail_poi_count_1000m",
        "restaurant_cafe_count_500m", "transit_stop_count_500m",
        "distance_to_nearest_rail_station", "distance_to_city_center",
    ]
    existing_meta = [c for c in station_meta_cols if c in merged.columns]

    if existing_meta:
        station_lookup = (
            demand.dropna(subset=existing_meta[:1])
            .groupby(["city", "station_key"])[existing_meta]
            .first()
            .reset_index()
        )
        # Drop old meta and re-merge
        merged.drop(columns=existing_meta, inplace=True, errors="ignore")
        merged = merged.merge(station_lookup, on=["city", "station_key"], how="left")

    # Fill weather from city-hour level
    weather_cols = [
        "temperature_2m", "relative_humidity_2m", "apparent_temperature",
        "precipitation", "rain", "snowfall", "cloud_cover", "wind_speed_10m",
    ]
    existing_weather = [c for c in weather_cols if c in demand.columns]

    if existing_weather:
        weather_lookup = (
            demand.dropna(subset=existing_weather[:1])
            .groupby(["city", "hour_ts"])[existing_weather]
            .first()
            .reset_index()
        )
        for col in existing_weather:
            if col in merged.columns:
                merged.drop(columns=[col], inplace=True, errors="ignore")
        merged = merged.merge(weather_lookup, on=["city", "hour_ts"], how="left")

    # Fill calendar from hour level
    calendar_cols = ["weekday", "weekend", "holiday", "working_day"]
    existing_cal = [c for c in calendar_cols if c in demand.columns]
    if existing_cal:
        cal_lookup = (
            demand.dropna(subset=existing_cal[:1])
            .groupby(["city", "hour_ts"])[existing_cal]
            .first()
            .reset_index()
        )
        for col in existing_cal:
            if col in merged.columns:
                merged.drop(columns=[col], inplace=True, errors="ignore")
        merged = merged.merge(cal_lookup, on=["city", "hour_ts"], how="left")

    # Fill remaining calendar NaNs from the timestamp
    if "weekday" in merged.columns:
        mask = merged["weekday"].isna()
        merged.loc[mask, "weekday"] = merged.loc[mask, "hour_ts"].dt.weekday
    if "weekend" in merged.columns:
        mask = merged["weekend"].isna()
        merged.loc[mask, "weekend"] = merged.loc[mask, "hour_ts"].dt.weekday.isin([5, 6]).astype(int)
    if "working_day" in merged.columns:
        mask = merged["working_day"].isna()
        merged.loc[mask, "working_day"] = (~merged.loc[mask, "hour_ts"].dt.weekday.isin([5, 6])).astype(int)
    if "holiday" in merged.columns:
        merged["holiday"] = merged["holiday"].fillna(0)

    return merged


def compute_historical_stats(demand_df: pd.DataFrame) -> dict:
    """Compute all historical aggregate features from training demand."""
    stats = {}

    # Station-level
    s = demand_df.groupby(["city", "station_key"])["demand"].agg(
        station_mean_demand="mean",
        station_median_demand="median",
        station_std_demand="std",
        station_total_rides="sum",
    ).reset_index()
    s["station_std_demand"] = s["station_std_demand"].fillna(0)
    stats["station_stats"] = s

    # Station-hour
    demand_df["hour"] = demand_df["hour_ts"].dt.hour
    demand_df["weekday_feat"] = demand_df["hour_ts"].dt.weekday

    sh = demand_df.groupby(["city", "station_key", "hour"])["demand"].agg(
        station_hour_mean="mean",
        station_hour_median="median",
    ).reset_index()
    stats["station_hour_stats"] = sh

    # Station-weekday
    sw = demand_df.groupby(["city", "station_key", "weekday_feat"])["demand"].agg(
        station_weekday_mean="mean",
    ).reset_index().rename(columns={"weekday_feat": "weekday"})
    stats["station_weekday_stats"] = sw

    # Station-weekday-hour
    swh = demand_df.groupby(["city", "station_key", "weekday_feat", "hour"])["demand"].agg(
        station_weekday_hour_mean="mean",
    ).reset_index().rename(columns={"weekday_feat": "weekday"})
    stats["station_weekday_hour_stats"] = swh

    # City-hour
    ch = demand_df.groupby(["city", "hour"])["demand"].agg(
        city_hour_mean="mean",
    ).reset_index()
    stats["city_hour_stats"] = ch

    # City-weekday-hour
    cwh = demand_df.groupby(["city", "weekday_feat", "hour"])["demand"].agg(
        city_weekday_hour_mean="mean",
    ).reset_index().rename(columns={"weekday_feat": "weekday"})
    stats["city_weekday_hour_stats"] = cwh

    # Global mean
    stats["global_mean_demand"] = float(demand_df["demand"].mean())

    return stats


def create_train_features(df: pd.DataFrame, city_map: dict) -> pd.DataFrame:
    """Build features for training (mirrors model.py _create_features)."""
    from model import BikeDemandModel
    # Use a temporary model instance to create features
    temp = BikeDemandModel()
    temp.artifacts = {
        "city_map": city_map,
        "global_mean_demand": 0.0,
        # Empty stats — we'll merge separately
        "station_stats": None,
        "station_hour_stats": None,
        "station_weekday_stats": None,
        "station_weekday_hour_stats": None,
        "city_hour_stats": None,
        "city_weekday_hour_stats": None,
    }

    # Prepare df in the format _create_features expects
    prep = df.copy()
    prep["start_station_id"] = prep["station_key"]

    # We need to provide the stats for merging
    return temp, prep


def main() -> None:
    print("Loading training data...")
    raw = pd.read_csv(TRAIN_CSV, low_memory=False)
    print(f"Loaded {len(raw):,} rides")

    # ---- Temporal split ----
    raw["_date"] = pd.to_datetime(raw["date"], dayfirst=True)
    all_dates = sorted(raw["_date"].unique())
    split_idx = int(len(all_dates) * 0.8)
    split_date = all_dates[split_idx]

    train_rides = raw[raw["_date"] < split_date].copy()
    print(f"Training rides: {len(train_rides):,} (before {split_date.date()})")

    # ---- Aggregate to station-hour demand ----
    print("Aggregating to station-hour demand...")
    demand = aggregate_demand(train_rides)
    print(f"Station-hour rows (observed): {len(demand):,}")

    # ---- Add zero-demand rows ----
    print("Adding zero-demand rows...")
    demand_full = add_zero_demand_rows(demand)
    print(f"Station-hour rows (with zeros): {len(demand_full):,}")

    # ---- Compute historical stats on FULL data (accurate averages) ----
    print("Computing historical aggregates...")
    hist_stats = compute_historical_stats(demand_full.copy())

    # ---- Downsample zero-demand rows for training ----
    # Keep all non-zero rows, sample 2x as many zero rows
    nonzero = demand_full[demand_full["demand"] > 0]
    zeros = demand_full[demand_full["demand"] == 0]
    n_zero_keep = min(len(zeros), len(nonzero) * 2)
    zeros_sampled = zeros.sample(n=n_zero_keep, random_state=42)
    demand_train = pd.concat([nonzero, zeros_sampled], ignore_index=True)
    print(f"Training rows after downsampling zeros: {len(demand_train):,} "
          f"({len(nonzero):,} non-zero + {n_zero_keep:,} zero)")

    # ---- City encoding ----
    cities = sorted(raw["city"].unique())
    city_map = {c: i for i, c in enumerate(cities)}

    # ---- Build feature matrix ----
    print("Building features...")
    from model import BikeDemandModel

    temp_model = BikeDemandModel()
    temp_model.artifacts = {
        "city_map": city_map,
        "global_mean_demand": hist_stats["global_mean_demand"],
        **hist_stats,
    }

    # Prepare for _create_features
    feat_df = demand_train.copy()
    feat_df["start_station_id"] = feat_df["station_key"]

    X = temp_model._create_features(feat_df)
    y = demand_train["demand"].values

    print(f"Feature matrix: {X.shape}")
    print(f"Target: mean={y.mean():.2f}, median={np.median(y):.2f}, max={y.max()}")

    # ---- Train LightGBM ----
    print("Training LightGBM...")
    params = {
        "objective": "mae",
        "metric": "mae",
        "boosting_type": "gbdt",
        "num_leaves": 255,
        "learning_rate": 0.05,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "min_child_samples": 30,
        "reg_alpha": 0.1,
        "reg_lambda": 0.1,
        "verbose": -1,
        "n_jobs": -1,
        "seed": 42,
    }

    train_data = lgb.Dataset(X, label=y)

    model = lgb.train(
        params,
        train_data,
        num_boost_round=1000,
    )

    # Print feature importance
    importance = model.feature_importance(importance_type="gain")
    feat_names = model.feature_name()
    top_features = sorted(zip(feat_names, importance), key=lambda x: -x[1])[:15]
    print("\nTop 15 features by gain:")
    for name, gain in top_features:
        print(f"  {name}: {gain:.0f}")

    # ---- Save artifacts ----
    artifacts = {
        "model": model,
        "city_map": city_map,
        **hist_stats,
    }

    joblib.dump(artifacts, OUTPUT_WEIGHTS)
    print(f"\nSaved {OUTPUT_WEIGHTS}")


if __name__ == "__main__":
    main()