# Energy_Consumption_Forecasting_LightGBM
This project demonstrates a complete workflow for forecasting energy consumption using machine learning (LightGBM)
# Electricity Consumption Forecasting with LightGBM

## Overview
This project is part of my journey in learning and exploring machine learning.  
I developed an end-to-end pipeline to forecast daily electricity consumption for multiple meters using real-world data.  
The pipeline demonstrates:
- feature engineering,
- model tuning,
- iterative multi-step forecasting,
- and evaluation.

AI-assisted coding tools were leveraged to explore best practices and accelerate development, while all modeling decisions, feature engineering, and result interpretations were conducted independently.

---

## Skills Highlighted
- **Time-series forecasting** with LightGBM  
- **Feature engineering**: lags, rolling statistics, differences, seasonal/cyclical, and interaction features  
- **Model tuning**: RandomizedSearchCV, TimeSeriesSplit  
- **Evaluation metrics**: RMSE, MAE, MAPE  
- **Iterative multi-step forecasting**  
- **Visualization & analysis** with Matplotlib and Pandas  
- **AI-assisted coding** to improve efficiency and maintain code quality  

---

## Project Steps

1. **Data Loading & Aggregation**  
   - Daily aggregation of raw meter readings from CSVs.  

2. **Feature Engineering**  
   - Lags and rolling statistics for each meter.  
   - Calendar & seasonal features.  
   - Interaction features (e.g., weekend * rolling mean).  
   - Optional: special events or holidays can be incorporated.  

3. **Model Training**  
   - LightGBM models trained per meter.  
   - Hyperparameter tuning with RandomizedSearchCV.  
   - TimeSeriesSplit cross-validation.  

4. **Evaluation**  
   - Metrics calculated for test set: RMSE, MAE, MAPE.  
   - Visual comparison of actual vs forecasted consumption.  

5. **Iterative Forecasting**  
   - Multi-day forecasts using predicted values for future steps.  

6. **Optional Clustering**  
   - Meters can be grouped by consumption profile to improve generalization.

---

