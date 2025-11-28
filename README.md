Impute PostValuation

This repository contains a small script to impute missing `PostValuation` values in `deals.parquet`.

Usage

- Install dependencies into the project virtualenv or system Python:

  /Users/c0p0frj/Documents/deal-valuation-imputer/.venv/bin/python -m pip install -r requirements.txt

- Run the imputer:

  /Users/c0p0frj/Documents/deal-valuation-imputer/.venv/bin/python scripts/impute_postvaluation.py

Outputs

- `deals_imputed.parquet` — same table with `PostValuation` filled where previously missing.

Notes & assumptions

- A RandomForestRegressor is trained on numeric and a few categorical features. The target is log1p transformed during training and inverse-transformed for predictions.
- This is a pragmatic baseline. Improvements can include better feature engineering, cross-validation, and using gradient boosting (LightGBM/XGBoost) or stacking.
