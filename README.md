## Imputation Utilities

This module provides functions for imputing missing target values in deal data.

### Functions

- `impute_missing_targets_rf(df, numerical, numerical_to_log, categorical, target_col)`
  - Imputes missing target values using a RandomForestRegressor.
  - Returns a DataFrame with imputed values and the trained model.

- `impute_target_by_similarity(row, df, categorical_cols, numerical_cols, range_dict, target_col)`
  - Imputes a missing target value for a row using similar rows in the DataFrame (based on categorical and numerical similarity).

### Example Usage

```python
from scripts.impute_postvaluation import impute_missing_targets_rf, impute_target_by_similarity

# Impute missing PostValuation using RandomForest
df_imputed, model = impute_missing_targets_rf(
    df,
    numerical=['DealSize'],
    numerical_to_log=['Revenue'],
    categorical=['Sector', 'Stage'],
    target_col='PostValuation'
)

# Impute a single row by similarity
imputed_value = impute_target_by_similarity(
    row,
    df,
    categorical_cols=['Sector', 'Stage'],
    numerical_cols=['DealSize'],
    range_dict={'DealSize': 0.2},
    target_col='PostValuation'
)