# Bike Demand Forecasting — Team Decisions

**Challenge:** Predict hourly bike demand per station for city1, city2 (main), city3/city4 (generalization).
**Metric:** MAE (lower is better). Negatives clipped to 0.
**Target:** `y_{s,d,h}` = # rides starting from station `s` during hour `h` on date `d`.

---

## Decisions

### Model Architecture
- **Hierarchical stat baseline**: 7-level fallback lookup (city+station+hour+weekday → global mean), stored as plain dicts for O(1) inference. Degrades gracefully for unseen cities (city4).
- **Tree model**: CatBoostRegressor (loss=MAE, 500 iterations, depth=6), trained on station-hour demand grid including explicit zero-demand rows (~55% of eval rows are zeros). City and station_id as native cat_features.
- **Features**: Cyclical sin/cos for hour (period=24) and weekday (period=7). `stat_baseline_pred` as a tree feature — tree refines the statistical prior.
- **Alpha blend**: final = α * tree + (1-α) * stat_baseline, α tuned on local validation set.
- **Training subset**: 50/50 nonzero/zero split, capped at N_ROWS_CAP=100k.

### MAE Results — Current Best (N_ROWS_CAP=100k, α=0.70)
| Model | Overall | city 1 | city 2 | city 3 |
|---|---|---|---|---|
| dummy_baseline | 98.91 | — | — | — |
| **my_team** | **0.764** | 0.981 | 0.626 | 0.365 |

Stat baseline alone: 0.776 | Tree alone: 0.767 | Best blend (α=0.70): 0.764

### What Didn't Work
- **N_ROWS_CAP 100k → 300k**: no improvement (0.764 both). Not a data volume problem.
- **recent_station_hour_mean**: data leakage — lookup included training rows themselves; tree overfit, val MAE → 0.872.
- **month feature**: only 2 unique values in training window; weak signal, MAE → 0.772.
