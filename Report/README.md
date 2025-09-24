# Electricity Consumption Forecasting with Machine Learning

## 1. Introduction
This notebook explores forecasting of daily electricity consumption using machine learning.  
The dataset originally had ~140,000 rows (15-minute intervals across ~340 meters), later aggregated to daily frequency (~1,462 rows).  

Two models were tested:  
- **LightGBM Regressor** (with hyperparameter tuning)  
- **Random Forest Regressor** (log-transformed targets)  

In addition, exploratory data analysis (EDA) and clustering were used to understand usage patterns across meters.  

---

## 2. EDA and Clustering
Meters were grouped by activity levels:  

- **Mostly Active (MA):** 331 meters  
- **Mostly Inactive (MI):** 39 meters  

Clustering on daily consumption revealed five active clusters plus one inactive group:  

| Cluster | Total Daily Consumption | Notes |
|---------|-------------------------|-------|
| 0 | 4,505 | Very low consumption |
| 4 | 42,728 | Low consumption |
| 2 | 497,519 | Medium consumption |
| 3 | 195,735 | Medium-low consumption |
| 1 | 901,974 | High consumption |
| -1 | 7,163 | Inactive group |

---

## 3. Forecasting Results
![MT_050_Actual_v.s_Forecast](figures/MT_050_Actual_v.s_Forecast.png) 
Sample results (LightGBM) for selected meters:  

| Meter | Zero % | RMSE | MAE | MAPE |
|-------|--------|------|-----|------|
| MT_202 | 0–10% | 736.6 | 258.0 | 32.96% |
| MT_180 | 0–10% | 678.1 | 304.4 | 58.97% |
| MT_132 | 10–20% | 117.3 | 77.6 | **9.92%** |
| MT_050 | 20–30% | 1182.0 | 284.6 | 37.01% |
| MT_173 | 30–40% | 312.7 | 79.5 | **68.59%** |
| MT_092 | 50–60% | 363.3 | 176.5 | 59.18% |
| MT_165 | 60–70% | 577.2 | 354.1 | **83.33%** |
| MT_120 | 70–80% | 501.4 | 325.0 | 32.05% |

---

## 4. Discussion
- **Zero Days Matter:** Forecast accuracy dropped sharply when meters had >50% zero days.  
- **Volatility Matters:** Even meters with few zeros (e.g., MT_202) showed errors if daily totals fluctuated strongly (100–300 → 30).  
- **Metrics Matter:** MAPE inflated for low-consumption meters; RMSE/NRMSE were more stable indicators.  

---
## 5. Conclusion
Forecasting electricity usage depends as much on data behavior as on the model itself.

**Key insights:**

- High-consumption, stable meters forecast well (MAPE ≈ 10%).

- Sparse or volatile meters remain challenging (MAPE 50–80%).

- Model comparison: LightGBM generally outperformed Random Forest, especially after hyperparameter tuning, though both struggled with highly volatile meters.

- Data aggregation: Due to computational constraints, the model was trained on daily aggregated data rather than 15-minute intervals. This aggregation is an important factor affecting performance, particularly for meters with rapid fluctuations.

**Future work:**

- Try alternative metrics (SMAPE, weighted MAPE).

- Build two-stage models (classify zero vs non-zero days, then forecast).

- Add external features (holidays, weather). 

**Future work:**  
- Try alternative metrics (SMAPE, weighted MAPE).  
- Build two-stage models (classify zero vs non-zero days, then forecast).  
- Add external features (holidays, weather).  

