# Bike Demand Forecasting — Decisions

**Metric:** MAE (lower is better). Negatives clipped to 0.

## Architecture
- **Stat baseline**: 7-level hierarchical mean lookup (city+station+hour+weekday → global mean). Fallback chain handles unseen cities (city4).
- **Tree**: CatBoostRegressor, loss=MAE, 500 iterations, depth=6, l2_leaf_reg=10. City and station_id as native cat_features.
- **Features**: sin/cos cyclical encoding for hour and weekday. `stat_baseline_pred` as a tree input feature.
- **Blend**: final = α·tree + (1-α)·stat_baseline, α tuned on validation set.
- **Training subset**: chronological — most recent N rows from demand grid (natural zero/nonzero ratio).

## Decisions
- Zero-demand rows are explicit in training (cross-product of stations × hours, unobserved = 0)
- Chronological sampling beats random 50/50 — more realistic distribution, recent data closest to validation
- More rows helps up to a point: 100k→300k→900k improved MAE but with diminishing returns
- Regularization (l2_leaf_reg) did not help — gap is distribution shift (Jan→Feb), not model complexity
- recent_station_hour_mean caused data leakage (lookup included training rows themselves)
- month feature too weak (only 2 unique values in training window)

## Trend Features — Tried and abandoned
- **Day-level trend** (OLS slope per city+station+hour+weekday): theory was sound but the signal was too noisy — only 6–7 training weeks, so slopes had high variance. Also had a global vs. per-city `train_end` bug (city 1 got negative weeks_ahead). After fixing the bug, MAE was still 0.768 — worse than the 0.751 baseline. City 2 (most rows, smallest weeks_ahead ≈ 0.14) regressed enough to override city 3's improvement.
- **Hour-level trend** (adjacent hour baselines): not attempted; dropped the day-level idea first.

## MAE Results
| Run | N_ROWS_CAP | Overall | city 1 | city 2 | city 3 |
|---|---|---|---|---|---|
| dummy baseline | — | 98.91 | — | — | — |
| random 50/50 | 100k | 0.764 | 0.981 | 0.626 | 0.365 |
| chronological | 300k | 0.761 | 0.980 | 0.621 | 0.364 |
| chronological | 900k | **0.751** | 0.978 | 0.607 | 0.347 |
| trend_pred (global bug→0.773; per-city fix→0.768) | 800k | 0.768 | 0.983 | 0.634 | 0.264 |
