"""
Imputation utilities.

Expose:
- `impute_missing_targets_rf(df, numerical, numerical_to_log, categorical, target_col)`
    Impute missing target values using RandomForestRegressor.
- `impute_target_by_similarity(row, df, categorical_cols, numerical_cols, range_dict, target_col)`
    Impute a missing target value for a row using similar rows in the DataFrame.
"""

from typing import List, Tuple, Dict, Any
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

def impute_missing_targets_rf(
    df: pd.DataFrame,
    numerical: List[str],
    numerical_to_log: List[str],
    categorical: List[str],
    target_col: str = 'target'
) -> Tuple[pd.DataFrame, RandomForestRegressor]:
    """
    Impute missing target values using RandomForestRegressor.

    Args:
        df: DataFrame containing features and target.
        numerical: List of column names to use as-is.
        numerical_to_log: List of column names to log-transform (log1p).
        categorical: List of categorical columns to one-hot encode.
        target_col: Name of the target column.

    Returns:
        df_imputed: DataFrame with missing target values filled (copy).
        model: Trained RandomForestRegressor.
    """
    df = df.copy()

    # Log-transform specified columns
    for col in numerical_to_log:
        if col in df.columns:
            df[col + '_log'] = np.log1p(df[col])
        else:
            raise ValueError(f"Column '{col}' not found in DataFrame.")

    # One-hot encode categoricals
    if categorical:
        df = pd.get_dummies(df, columns=categorical, drop_first=True)

    # Build feature column list
    logcols = [col + '_log' for col in numerical_to_log]
    categorical_expanded = [c for c in df.columns if any(c.startswith(cat + '_') for cat in categorical)]
    feature_cols = numerical + logcols + categorical_expanded

    # Ensure all feature columns exist
    missing_features = [col for col in feature_cols if col not in df.columns]
    if missing_features:
        raise ValueError(f"Missing feature columns: {missing_features}")

    mask_known = df[target_col].notna()
    X_known = df.loc[mask_known, feature_cols].values
    y_known = df.loc[mask_known, target_col].values

    # Train/test split
    X_train, X_val, y_train, y_val = train_test_split(X_known, y_known, test_size=0.2, random_state=5)

    model = RandomForestRegressor(n_estimators=100, random_state=1, n_jobs=-1)
    model.fit(X_train, y_train)

    # Metrics
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)
    train_rmse = float(np.sqrt(mean_squared_error(y_train, y_train_pred)))
    val_rmse = float(np.sqrt(mean_squared_error(y_val, y_val_pred)))
    train_r2 = float(r2_score(y_train, y_train_pred))
    val_r2 = float(r2_score(y_val, y_val_pred))

    setattr(model, 'train_rmse', train_rmse)
    setattr(model, 'test_rmse', val_rmse)
    setattr(model, 'train_r2', train_r2)
    setattr(model, 'test_r2', val_r2)

    print(f"Train RMSE: {train_rmse:.4f}, Train R2: {train_r2:.4f}")
    print(f"Test  RMSE: {val_rmse:.4f}, Test R2: {val_r2:.4f}")

    # Impute missing
    mask_missing = ~mask_known
    if mask_missing.sum() > 0:
        X_missing = df.loc[mask_missing, feature_cols].values
        y_pred = model.predict(X_missing)
        df.loc[mask_missing, target_col] = y_pred

    # Add prediction column
    df[target_col + '_predicted'] = model.predict(df[feature_cols].values)

    return df, model

def impute_target_by_similarity(
    row: pd.Series,
    df: pd.DataFrame,
    categorical_cols: List[str],
    numerical_cols: List[str],
    range_dict: Dict[str, float],
    target_col: str = 'target'
) -> Any:
    """
    Impute missing target value for a row using similar rows in the DataFrame.

    Args:
        row: The row to impute (Series).
        df: The full DataFrame.
        categorical_cols: List of categorical column names to match exactly.
        numerical_cols: List of numerical column names to match within range.
        range_dict: Dict mapping numerical column names to allowed percentage range (e.g., {'DealSize': 0.2}).
        target_col: The column to impute.

    Returns:
        Imputed value (median of similar rows), or np.nan if no similar rows found.
    """
    mask = pd.Series(True, index=df.index)

    # Categorical: exact match
    for col in categorical_cols:
        if col in df.columns:
            val = row[col]
            if pd.isnull(val):
                mask &= df[col].isnull()
            else:
                mask &= (df[col] == val)
        else:
            raise ValueError(f"Categorical column '{col}' not found in DataFrame.")
        
    # Numerical: within range
    for col in numerical_cols:
        if col in df.columns:
            val = row[col]
            rng = range_dict.get(col, 0.80)
            if pd.isnull(val):
                mask &= df[col].isnull()
            else:
                lower = val * (1 - rng)
                upper = val * (1 + rng)
                mask &= df[col].between(lower, upper)
        else:
            raise ValueError(f"Numerical column '{col}' not found in DataFrame.")
    
    # # Numerical: within range
    # for col in numerical_cols:
    #     if col in df.columns:
    #         val = row[col]
    #         rng = range_dict.get(col, 0.1)
    #         lower = val * (1 - rng)
    #         upper = val * (1 + rng)
    #         mask &= df[col].between(lower, upper)
    #     else:
    #         raise ValueError(f"Numerical column '{col}' not found in DataFrame.")

    # Exclude the row itself and rows with missing target
    mask &= df[target_col].notnull()
    mask &= df.index != row.name

    similar = df[mask]
    if not similar.empty:
        return similar[target_col].median()
    else:
        return np.nan

# Example usage:
# df_imputed, model = impute_missing_targets_rf(df, ['DealSize'], ['Revenue'], ['Sector', 'Stage'], target_col='PostValuation')
# imputed_value = impute_target_by_similarity(row, df, ['Sector', 'Stage'], ['DealSize'], {'DealSize': 0.2}, target_col='PostValuation')