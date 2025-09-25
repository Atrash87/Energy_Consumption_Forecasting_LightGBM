"""
LightGBM time-series pipeline for energy meters (daily aggregation).
- Per-meter or per-cluster training
- Feature engineering: lags, rolling stats, seasonal, diffs, calendar
- TimeSeriesSplit + RandomizedSearchCV tuning
- Log-transform target (log1p)
- Iterative multi-day forecasting

This file contains the function library. Use run_pipeline.py to execute the pipeline.
"""

import os
import warnings
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error, mean_absolute_error
from lightgbm import LGBMRegressor
import joblib

RANDOM_STATE = 42
warnings.filterwarnings("ignore")


# --------------------------- Data loading & aggregation --------------------------- #
def load_and_aggregate(file_path: str,
                       datetime_col: str = "DateTime",
                       freq: str = "D",
                       agg_method: str = "sum") -> pd.DataFrame:
    """
    Load CSV with a datetime column and multiple meter columns, set datetime index and aggregate.
    - freq: pandas resample frequency, e.g. 'D' for daily
    - agg_method: 'sum' or 'mean'
    Returns df_agg with DatetimeIndex and meter columns.
    """
    print("Loading CSV:", file_path)
    df = pd.read_csv(file_path, parse_dates=[datetime_col])
    df = df.set_index(datetime_col).sort_index()
    if agg_method == "sum":
        df_agg = df.resample(freq).sum()
    elif agg_method == "mean":
        df_agg = df.resample(freq).mean()
    else:
        raise ValueError("agg_method must be 'sum' or 'mean'")
    print(f"Loaded {df.shape[0]} rows, aggregated to {df_agg.shape[0]} rows ({freq}).")
    return df_agg


# --------------------------- Feature engineering --------------------------- #
def make_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add calendar and seasonal features common to all meters"""
    idx = df.index
    out = pd.DataFrame(index=idx)
    out["dayofweek"] = idx.dayofweek
    out["month"] = idx.month
    out["quarter"] = idx.quarter
    out["year"] = idx.year
    out["dayofyear"] = idx.dayofyear
    out["is_weekend"] = (idx.dayofweek >= 5).astype(int)
    out["is_month_start"] = idx.is_month_start.astype(int)
    out["is_month_end"] = idx.is_month_end.astype(int)
    # seasonal circular encodings
    out["sin_dayofyear"] = np.sin(2 * np.pi * out["dayofyear"] / 365.25)
    out["cos_dayofyear"] = np.cos(2 * np.pi * out["dayofyear"] / 365.25)
    out["sin_month"] = np.sin(2 * np.pi * out["month"] / 12)
    out["cos_month"] = np.cos(2 * np.pi * out["month"] / 12)
    return out


def build_features_for_series(series: pd.Series,
                              max_lag: int = 365,
                              lags: List[int] = [1, 2, 3, 7, 14, 30, 365],
                              rolling_windows: List[int] = [7, 14, 30, 90]) -> pd.DataFrame:
    """
    Build features for a single series (one meter) and returns DataFrame of features + target.
    Drops NaNs so ready for training.
    """
    df = pd.DataFrame({"target": series}).copy()
    df = df.sort_index()
    # calendar & seasonal features
    time_feats = make_time_features(df)
    df = pd.concat([df, time_feats], axis=1)

    # lags
    for lag in lags:
        df[f"lag_{lag}"] = df["target"].shift(lag)

    # rolling stats
    for w in rolling_windows:
        df[f"roll_mean_{w}"] = df["target"].rolling(window=w, min_periods=1).mean()
        df[f"roll_std_{w}"] = df["target"].rolling(window=w, min_periods=1).std().fillna(0)
        df[f"roll_min_{w}"] = df["target"].rolling(window=w, min_periods=1).min()
        df[f"roll_max_{w}"] = df["target"].rolling(window=w, min_periods=1).max()

    # diffs
    df["diff_1"] = df["target"].diff(1)
    df["diff_7"] = df["target"].diff(7)
    df["diff_30"] = df["target"].diff(30)

    # drop rows with NaN (due to lags)
    df = df.dropna()
    return df


# --------------------------- Clustering meters --------------------------- #
def cluster_meters_by_profile(df_agg: pd.DataFrame, n_clusters: int = 4) -> Dict[int, List[str]]:
    """
    Cluster meters by mean daily profile across the whole period.
    Returns dict cluster_id -> list of meter column names.
    """
    print(f"Clustering {df_agg.shape[1]} meters into {n_clusters} clusters...")
    # Use normalized mean profile per meter
    meter_means = df_agg.mean(axis=0).values.reshape(-1, 1)
    kmeans = KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE)
    labels = kmeans.fit_predict(meter_means)
    clusters = {}
    cols = df_agg.columns.tolist()
    for col, lbl in zip(cols, labels):
        clusters.setdefault(int(lbl), []).append(col)
    for k, v in clusters.items():
        print(f" Cluster {k}: {len(v)} meters")
    return clusters


# --------------------------- Hyperparameter tuning for LightGBM --------------------------- #
def tune_lgbm(X: np.ndarray, y: np.ndarray, n_iter: int = 20) -> LGBMRegressor:
    """
    RandomizedSearchCV with TimeSeriesSplit to tune LightGBM regressor.
    Returns best estimator (already fitted on the CV folds inside search).
    """
    param_dist = {
        "num_leaves": [31, 50, 80, 100, 150],
        "max_depth": [-1, 6, 10, 20, 30],
        "learning_rate": [0.01, 0.03, 0.05, 0.1],
        "n_estimators": [200, 500, 800, 1200],
        "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
        "colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
        "reg_alpha": [0.0, 0.1, 0.5],
        "reg_lambda": [0.0, 0.1, 0.5],
    }
    base = LGBMRegressor(random_state=RANDOM_STATE, n_jobs=-1)
    tscv = TimeSeriesSplit(n_splits=3)
    search = RandomizedSearchCV(base, param_distributions=param_dist, n_iter=n_iter,
                                cv=tscv, scoring="neg_mean_squared_error",
                                random_state=RANDOM_STATE, n_jobs=-1, verbose=0)
    print("Starting RandomizedSearchCV for LightGBM...")
    search.fit(X, y)
    print("Best params:", search.best_params_)
    print("Best CV neg_mse:", search.best_score_)
    return search.best_estimator_


# --------------------------- Training per-meter or per-cluster --------------------------- #
def train_models(df_agg: pd.DataFrame,
                 meters: Optional[List[str]] = None,
                 use_clustering: bool = False,
                 n_clusters: int = 4,
                 test_size: float = 0.2,
                 tune: bool = True,
                 tune_iter: int = 20,
                 output_dir: str = "models") -> Dict[str, dict]:
    """
    Train LightGBM models.
    - If use_clustering=False, trains one model per meter in meters list (or all columns if meters=None).
    - If use_clustering=True, clusters meters and trains one model per cluster using aggregated series (mean of meters in cluster).
    Returns dict of models metadata keyed by model name.
    Models & scalers saved under output_dir.
    """
    os.makedirs(output_dir, exist_ok=True)
    if meters is None:
        meters = df_agg.columns.tolist()

    models_info = {}

    if use_clustering:
        clusters = cluster_meters_by_profile(df_agg, n_clusters=n_clusters)
        training_groups = {f"cluster_{k}": df_agg[cols].mean(axis=1) for k, cols in clusters.items()}
    else:
        training_groups = {m: df_agg[m] for m in meters}

    for name, series in training_groups.items():
        print("\n======================================")
        print("Training model for:", name)
        df_feats = build_features_for_series(series)
        # chronological split
        cutoff = int(len(df_feats) * (1 - test_size))
        train = df_feats.iloc[:cutoff]
        test = df_feats.iloc[cutoff:]
        X_train = train.drop(columns=["target"])
        y_train = np.log1p(train["target"])
        X_test = test.drop(columns=["target"])
        y_test = test["target"]

        # scale features
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        # tuning
        if tune:
            model = tune_lgbm(X_train_s, y_train, n_iter=tune_iter)
        else:
            model = LGBMRegressor(random_state=RANDOM_STATE, n_jobs=-1, n_estimators=500, learning_rate=0.05)

        # fit final model on full training set (X_train_s is already used for CV during tuning;
        # we re-fit to ensure model trained on training set only)
        model.fit(X_train_s, y_train)

        # predict and inverse transform
        y_pred_log = model.predict(X_test_s)
        y_pred = np.expm1(y_pred_log)

        # metrics
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        # avoid divide by zero in mape
        mask = y_test != 0
        if mask.sum() > 0:
            mape = (np.abs((y_test[mask] - y_pred[mask]) / y_test[mask])).mean() * 100
        else:
            mape = np.nan

        print(f"Test RMSE: {rmse:.3f} | MAE: {mae:.3f} | MAPE: {mape:.2f}%")

        # save model + scaler + feature columns
        model_fname = os.path.join(output_dir, f"{name}_lgbm.joblib")
        scaler_fname = os.path.join(output_dir, f"{name}_scaler.joblib")
        meta = {
            "model_name": name,
            "model": model,
            "scaler": scaler,
            "feature_columns": X_train.columns.tolist(),
            "train_index": train.index,
            "test_index": test.index,
            "metrics": {"rmse": rmse, "mae": mae, "mape": mape},
            "train_last_target": train["target"].iloc[-1]
        }
        joblib.dump(model, model_fname)
        joblib.dump(scaler, scaler_fname)
        # optional: dump meta summary (without heavy objects)
        meta_small = {k: v for k, v in meta.items() if k not in ["model", "scaler"]}
        joblib.dump(meta_small, os.path.join(output_dir, f"{name}_meta.joblib"))

        models_info[name] = meta
        # store predictions DataFrame for quick inspection
        df_test = test.copy()
        df_test["forecast"] = y_pred
        models_info[name]["test_df"] = df_test

        # quick plot
        try:
            plt.figure(figsize=(12, 4))
            plt.plot(df_test.index, df_test["target"], label="Actual", linewidth=2)
            plt.plot(df_test.index, df_test["forecast"], label="Forecast", linewidth=1.5)
            plt.title(f"{name} - Actual vs Forecast")
            plt.legend()
            plt.tight_layout()
            plt.show()
        except Exception:
            pass

    print("\nAll models trained and saved to:", output_dir)
    return models_info


# --------------------------- Forecast future iteratively --------------------------- #
def create_future_row(date: pd.Timestamp, historical_df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    """
    Create a single feature-row (1 x n_features) for future date using last historical_df containing 'target'.
    Assumes historical_df index is datetime and contains a 'target' column with actuals/predictions.
    """
    row = {}
    # calendar & seasonal features
    row["dayofweek"] = date.dayofweek
    row["month"] = date.month
    row["quarter"] = date.quarter
    row["year"] = date.year
    row["dayofyear"] = date.dayofyear
    row["is_weekend"] = 1 if date.dayofweek >= 5 else 0
    row["is_month_start"] = 1 if date.is_month_start else 0
    row["is_month_end"] = 1 if date.is_month_end else 0
    row["sin_dayofyear"] = np.sin(2 * np.pi * date.dayofyear / 365.25)
    row["cos_dayofyear"] = np.cos(2 * np.pi * date.dayofyear / 365.25)
    row["sin_month"] = np.sin(2 * np.pi * date.month / 12)
    row["cos_month"] = np.cos(2 * np.pi * date.month / 12)

    # fill lag and rolling features by looking back into historical_df['target']
    # helpers
    last = historical_df["target"]
    for col in feature_cols:
        if col.startswith("lag_"):
            lag = int(col.split("_")[-1])
            if len(last) >= lag:
                row[col] = last.iloc[-lag]
            else:
                row[col] = last.iloc[-1]
        elif col.startswith("roll_mean_"):
            w = int(col.split("_")[-1])
            row[col] = last.rolling(w, min_periods=1).mean().iloc[-1]
        elif col.startswith("roll_std_"):
            w = int(col.split("_")[-1])
            row[col] = last.rolling(w, min_periods=1).std().fillna(0).iloc[-1]
        elif col.startswith("roll_min_"):
            w = int(col.split("_")[-1])
            row[col] = last.rolling(w, min_periods=1).min().iloc[-1]
        elif col.startswith("roll_max_"):
            w = int(col.split("_")[-1])
            row[col] = last.rolling(w, min_periods=1).max().iloc[-1]
        elif col.startswith("diff_"):
            d = int(col.split("_")[-1])
            if len(last) >= d + 1:
                row[col] = last.iloc[-1] - last.iloc[-d - 1]
            else:
                row[col] = 0
        elif col in ["dayofweek", "month", "quarter", "year", "dayofyear",
                     "is_weekend", "is_month_start", "is_month_end",
                     "sin_dayofyear", "cos_dayofyear", "sin_month", "cos_month"]:
            # already set above
            pass
        else:
            # unknown feature: fill with last value or 0
            row[col] = last.iloc[-1] if "lag" in col or "roll" in col or "diff" in col else 0

    # ensure ordering
    row_df = pd.DataFrame([row], index=[date])[feature_cols]
    return row_df


def forecast_iterative(model: LGBMRegressor,
                       scaler: StandardScaler,
                       historical_df: pd.DataFrame,
                       feature_cols: List[str],
                       steps: int = 30) -> pd.DataFrame:
    """
    Iteratively forecast `steps` days into the future.
    - historical_df: a DataFrame including a 'target' column up to the last observed date
    - feature_cols: list of columns in the same order used during training
    Returns DataFrame index=dates, column 'forecast'
    """
    hist = historical_df.copy()
    preds = []
    last_date = hist.index[-1]
    for i in range(steps):
        next_date = last_date + pd.Timedelta(days=1)
        # create features row using hist['target']
        row = create_future_row(next_date, hist, feature_cols)
        X_row_scaled = scaler.transform(row.values)  # scaler expects 2D array
        y_pred_log = model.predict(X_row_scaled)[0]
        y_pred = np.expm1(y_pred_log)
        preds.append((next_date, y_pred))
        # append to hist so next day can use predicted target for lags/rolls
        hist = pd.concat([hist, pd.DataFrame({"target": [y_pred]}, index=[next_date])])
        last_date = next_date
    df_preds = pd.DataFrame(preds, columns=["date", "forecast"]).set_index("date")
    return df_preds