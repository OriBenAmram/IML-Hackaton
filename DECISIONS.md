# Bike Demand Forecasting — Team Decisions

**Challenge:** Predict hourly bike demand per station for city1, city2 (main), city3/city4 (generalization).
**Metric:** MAE (lower is better). Negatives clipped to 0.
**Target:** `y_{s,d,h}` = # rides starting from station `s` during hour `h` on date `d`.

---

## Model Architecture
- **Hierarchical stat baseline**: 7-level fallback lookup (city+station+hour+weekday → global mean), stored as plain dicts for O(1) inference. Degrades gracefully for unseen cities (city4).
- **Tree model**: CatBoostRegressor (loss=MAE, 500 iterations, depth=6), trained on station-hour demand grid including explicit zero-demand rows. City and station_id as native cat_features.
- **Features**: Cyclical sin/cos for hour (period=24) and weekday (period=7). `stat_baseline_pred` as a tree feature — tree refines the statistical prior.
- **Alpha blend**: final = α * tree + (1-α) * stat_baseline, α tuned on local validation set.
- **Training subset**: Chronological — most recent N_ROWS_CAP rows by hour_ts (natural zero/nonzero ratio, closest to validation period). Total grid: 1,371,536 rows; currently using 300k (1,071,536 remaining).

## MAE Results

| Run | N_ROWS_CAP | Sampling | Overall MAE | city 1 | city 2 | city 3 |
|---|---|---|---|---|---|---|
| dummy baseline | — | — | 98.91 | — | — | — |
| v1 | 100k | random 50/50 | 0.764 | 0.981 | 0.626 | 0.365 |
| v2 | 100k | chronological | 0.776 | 0.975 | 0.651 | 0.334 |
| **v3 (current)** | **300k** | **chronological** | **0.761** | **0.980** | **0.621** | **0.364** |

## What Didn't Work
- **Random 300k (50/50)**: same as 100k (0.764). Not a data volume problem with that sampling.
- **recent_station_hour_mean**: data leakage — lookup included training rows; overfit, MAE → 0.872.
- **month feature**: only 2 unique values in training; weak signal, MAE → 0.772.
- **Chronological 100k**: tree saw only 17k nonzero examples (83% zeros), barely contributed (alpha=0.10), MAE → 0.776.
