import numpy as np
import pandas as pd


class BikeDemandModel:
    """
    Bike-demand prediction model using LightGBM.

    This class handles:
        - Feature engineering from station-hour target rows
        - Merging pre-computed historical aggregates
        - Prediction using a trained LightGBM model
    """

    # Features used by the model (must match training)
    FEATURE_COLUMNS = [
        # --- Temporal ---
        "hour",
        "weekday",
        "month",
        "day_of_month",
        "weekend",
        "holiday",
        "working_day",
        "is_rush_morning",
        "is_rush_evening",
        "is_rush_hour",
        "is_night",
        "is_midday",
        "hour_sin",
        "hour_cos",
        "weekday_sin",
        "weekday_cos",
        "month_sin",
        "month_cos",
        # --- Weather ---
        "temperature_2m",
        "relative_humidity_2m",
        "apparent_temperature",
        "precipitation",
        "rain",
        "snowfall",
        "cloud_cover",
        "wind_speed_10m",
        "is_rainy",
        "is_snowy",
        "is_cold",
        "is_hot",
        "temp_feels_diff",
        # --- Station metadata ---
        "start_lat",
        "start_lng",
        "bike_lane_length_500m",
        "park_area_500m",
        "university_count_1000m",
        "office_poi_count_1000m",
        "retail_poi_count_1000m",
        "restaurant_cafe_count_500m",
        "transit_stop_count_500m",
        "distance_to_nearest_rail_station",
        "distance_to_city_center",
        # --- Interactions ---
        "office_x_rush",
        "university_x_workday",
        "park_x_weekend",
        "park_x_weekend_x_temp",
        "restaurant_x_evening",
        "transit_x_rush",
        "temp_x_weekend",
        # --- Historical aggregates ---
        "station_mean_demand",
        "station_median_demand",
        "station_std_demand",
        "station_hour_mean",
        "station_hour_median",
        "station_weekday_mean",
        "station_weekday_hour_mean",
        "city_hour_mean",
        "city_weekday_hour_mean",
        "station_total_rides",
        # --- City encoding ---
        "city_encoded",
    ]

    def __init__(self):
        self.artifacts = None

    def load_artifacts(self, artifacts: dict) -> None:
        """
        Store all objects created by train.py.
        """
        self.artifacts = artifacts

    def _create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Build features from a station-hour target dataframe.

        This must produce the exact same features used during training.
        """
        out = df.copy()

        # ---- Parse timestamps ----
        # The evaluator provides hour_ts or target_hour_start as strings
        if "hour_ts" in out.columns:
            ts = pd.to_datetime(out["hour_ts"], errors="coerce")
        elif "target_hour_start" in out.columns:
            ts = pd.to_datetime(out["target_hour_start"], errors="coerce")
        else:
            raise ValueError("Need hour_ts or target_hour_start column")

        # ---- Temporal features ----
        out["hour"] = ts.dt.hour
        out["weekday"] = ts.dt.weekday
        out["month"] = ts.dt.month
        out["day_of_month"] = ts.dt.day

        # Ensure weekend/holiday/working_day exist
        if "weekend" not in out.columns:
            out["weekend"] = out["weekday"].isin([5, 6]).astype(int)
        if "holiday" not in out.columns:
            out["holiday"] = 0
        if "working_day" not in out.columns:
            out["working_day"] = (~out["weekday"].isin([5, 6])).astype(int)

        out["is_rush_morning"] = out["hour"].isin([7, 8, 9]).astype(int)
        out["is_rush_evening"] = out["hour"].isin([17, 18, 19]).astype(int)
        out["is_rush_hour"] = (out["is_rush_morning"] | out["is_rush_evening"]).astype(int)
        out["is_night"] = out["hour"].isin([22, 23, 0, 1, 2, 3, 4, 5]).astype(int)
        out["is_midday"] = out["hour"].isin([11, 12, 13]).astype(int)

        # Cyclical encoding
        out["hour_sin"] = np.sin(2 * np.pi * out["hour"] / 24)
        out["hour_cos"] = np.cos(2 * np.pi * out["hour"] / 24)
        out["weekday_sin"] = np.sin(2 * np.pi * out["weekday"] / 7)
        out["weekday_cos"] = np.cos(2 * np.pi * out["weekday"] / 7)
        out["month_sin"] = np.sin(2 * np.pi * out["month"] / 12)
        out["month_cos"] = np.cos(2 * np.pi * out["month"] / 12)

        # ---- Weather features ----
        for col in ["temperature_2m", "relative_humidity_2m", "apparent_temperature",
                     "precipitation", "rain", "snowfall", "cloud_cover", "wind_speed_10m"]:
            if col not in out.columns:
                out[col] = np.nan

        out["is_rainy"] = (out["precipitation"].fillna(0) > 0).astype(int)
        out["is_snowy"] = (out["snowfall"].fillna(0) > 0).astype(int)
        out["is_cold"] = (out["temperature_2m"].fillna(15) < 5).astype(int)
        out["is_hot"] = (out["temperature_2m"].fillna(15) > 35).astype(int)
        out["temp_feels_diff"] = (
            out["temperature_2m"].fillna(0) - out["apparent_temperature"].fillna(0)
        )

        # ---- Station metadata ----
        station_cols = [
            "start_lat", "start_lng", "bike_lane_length_500m", "park_area_500m",
            "university_count_1000m", "office_poi_count_1000m", "retail_poi_count_1000m",
            "restaurant_cafe_count_500m", "transit_stop_count_500m",
            "distance_to_nearest_rail_station", "distance_to_city_center",
        ]
        for col in station_cols:
            if col not in out.columns:
                out[col] = np.nan

        # ---- Interaction features ----
        out["office_x_rush"] = out["office_poi_count_1000m"].fillna(0) * out["is_rush_hour"]
        out["university_x_workday"] = out["university_count_1000m"].fillna(0) * out["working_day"]
        out["park_x_weekend"] = out["park_area_500m"].fillna(0) * out["weekend"]
        out["park_x_weekend_x_temp"] = (
            out["park_area_500m"].fillna(0) * out["weekend"] * out["temperature_2m"].fillna(15)
        )
        out["restaurant_x_evening"] = (
            out["restaurant_cafe_count_500m"].fillna(0) * out["is_midday"]
        )
        out["transit_x_rush"] = out["transit_stop_count_500m"].fillna(0) * out["is_rush_hour"]
        out["temp_x_weekend"] = out["temperature_2m"].fillna(15) * out["weekend"]

        # ---- City encoding ----
        city_map = self.artifacts.get("city_map", {})
        if "city" in out.columns:
            out["city_encoded"] = out["city"].map(city_map).fillna(-1).astype(int)
        else:
            out["city_encoded"] = -1

        # ---- Normalize station ID for merging ----
        if "start_station_id" in out.columns:
            out["_station_key"] = out["start_station_id"].astype(str).str.strip()
            # Handle float-like IDs: "3074.0" -> "3074"
            numeric = pd.to_numeric(out["_station_key"], errors="coerce")
            is_int = numeric.notna() & np.isfinite(numeric) & (numeric % 1 == 0)
            out.loc[is_int, "_station_key"] = numeric.loc[is_int].astype(int).astype(str)
        else:
            out["_station_key"] = "__missing__"

        if "city" in out.columns:
            out["_city_key"] = out["city"].astype(str)
        else:
            out["_city_key"] = "__missing__"

        # ---- Merge historical aggregates ----
        # Station-level stats
        station_stats = self.artifacts.get("station_stats")
        if station_stats is not None:
            out = out.merge(
                station_stats,
                left_on=["_city_key", "_station_key"],
                right_on=["city", "station_key"],
                how="left",
                suffixes=("", "_stat"),
            )
            # Drop merge keys from stats table
            out.drop(columns=["city_stat", "station_key"], errors="ignore", inplace=True)

        # Station-hour stats
        station_hour_stats = self.artifacts.get("station_hour_stats")
        if station_hour_stats is not None:
            out = out.merge(
                station_hour_stats,
                left_on=["_city_key", "_station_key", "hour"],
                right_on=["city", "station_key", "hour"],
                how="left",
                suffixes=("", "_sh"),
            )
            out.drop(columns=["city_sh", "station_key_sh"], errors="ignore", inplace=True)

        # Station-weekday stats
        station_weekday_stats = self.artifacts.get("station_weekday_stats")
        if station_weekday_stats is not None:
            out = out.merge(
                station_weekday_stats,
                left_on=["_city_key", "_station_key", "weekday"],
                right_on=["city", "station_key", "weekday"],
                how="left",
                suffixes=("", "_sw"),
            )
            out.drop(columns=["city_sw", "station_key_sw"], errors="ignore", inplace=True)

        # Station-weekday-hour stats
        station_wh_stats = self.artifacts.get("station_weekday_hour_stats")
        if station_wh_stats is not None:
            out = out.merge(
                station_wh_stats,
                left_on=["_city_key", "_station_key", "weekday", "hour"],
                right_on=["city", "station_key", "weekday", "hour"],
                how="left",
                suffixes=("", "_swh"),
            )
            out.drop(columns=["city_swh", "station_key_swh"], errors="ignore", inplace=True)

        # City-hour stats
        city_hour_stats = self.artifacts.get("city_hour_stats")
        if city_hour_stats is not None:
            out = out.merge(
                city_hour_stats,
                left_on=["_city_key", "hour"],
                right_on=["city", "hour"],
                how="left",
                suffixes=("", "_ch"),
            )
            out.drop(columns=["city_ch"], errors="ignore", inplace=True)

        # City-weekday-hour stats
        city_wh_stats = self.artifacts.get("city_weekday_hour_stats")
        if city_wh_stats is not None:
            out = out.merge(
                city_wh_stats,
                left_on=["_city_key", "weekday", "hour"],
                right_on=["city", "weekday", "hour"],
                how="left",
                suffixes=("", "_cwh"),
            )
            out.drop(columns=["city_cwh"], errors="ignore", inplace=True)

        # ---- Fill missing aggregate features with global fallback ----
        global_mean = self.artifacts.get("global_mean_demand", 0.0)
        agg_cols = [
            "station_mean_demand", "station_median_demand", "station_std_demand",
            "station_hour_mean", "station_hour_median",
            "station_weekday_mean", "station_weekday_hour_mean",
            "city_hour_mean", "city_weekday_hour_mean", "station_total_rides",
        ]
        for col in agg_cols:
            if col not in out.columns:
                out[col] = global_mean
            else:
                out[col] = out[col].fillna(global_mean)

        # ---- Select final feature columns ----
        # Make sure all columns exist
        for col in self.FEATURE_COLUMNS:
            if col not in out.columns:
                out[col] = 0

        return out[self.FEATURE_COLUMNS]

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

        model = self.artifacts["model"]
        X = self._create_features(test_df)
        preds = model.predict(X)

        # Bike demand cannot be negative.
        return np.maximum(0.0, preds)