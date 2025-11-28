"""Minimal imputer utility.

Expose a single function `impute_df_basic(df, feature_cols, target_col)` which:
 - trains a RandomForestRegressor on rows where target is present using only the provided feature columns
 - predicts the target for rows where it is missing
 - returns (df_imputed, model)

This module purposely performs no normalization, encoding, or feature engineering.
The caller is responsible for preparing numeric arrays (e.g., encoding categoricals) if needed.
"""
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


def impute_df_basic(df: pd.DataFrame, feature_cols: List[str], target_col: str = 'PostValuation') -> Tuple[pd.DataFrame, RandomForestRegressor]:
    """Train a RandomForest on given features and impute missing target values.

    Args:
        df: DataFrame containing feature columns and target.
        feature_cols: list of column names to use as features. These columns must be numeric (caller responsibility).
        target_col: name of the target column.

    Returns:
        df_imputed: DataFrame with missing target values filled in-place (a copy is returned).
        model: trained RandomForestRegressor (or None if no training occurred).
    """
    df = df.copy()

    mask_known = df[target_col].notna()

    X_known = df.loc[mask_known, feature_cols].values
    y_known = df.loc[mask_known, target_col].values

    # split known data to estimate train/test error
    X_tr, X_val, y_tr, y_val = train_test_split(X_known, y_known, test_size=0.2, random_state=1)

    model = RandomForestRegressor(n_estimators=100, random_state=1, n_jobs=-1)
    model.fit(X_tr, y_tr)

    # compute metrics
    y_tr_pred = model.predict(X_tr)
    y_val_pred = model.predict(X_val)
    train_rmse = float(np.sqrt(mean_squared_error(y_tr, y_tr_pred)))
    val_rmse = float(np.sqrt(mean_squared_error(y_val, y_val_pred)))
    train_r2 = float(r2_score(y_tr, y_tr_pred))
    val_r2 = float(r2_score(y_val, y_val_pred))

    # attach metrics to model for caller access
    setattr(model, 'train_rmse', train_rmse)
    setattr(model, 'test_rmse', val_rmse)
    setattr(model, 'train_r2', train_r2)
    setattr(model, 'test_r2', val_r2)

    print(f"Train RMSE: {train_rmse:.4f}, Train R2: {train_r2:.4f}")
    print(f"Test  RMSE: {val_rmse:.4f}, Test R2: {val_r2:.4f}")

    # predict missing
    mask_missing = ~mask_known
    if mask_missing.sum() > 0:
        X_missing = df.loc[mask_missing, feature_cols].values
        y_pred = model.predict(X_missing)
        df.loc[mask_missing, target_col] = y_pred

    df['PostValuation_predicted'] = model.predict(df[feature_cols].values)


    return df, model
