#!/usr/bin/env python3
import sys
import pandas as pd

def main():
    p = 'deals_imputed.parquet'
    df = pd.read_parquet(p)
    nulls = df['PostValuation'].isna().sum()
    print('rows', len(df))
    print('PostValuation nulls:', nulls)
    if nulls != 0:
        print('Validation FAILED: nulls remain')
        sys.exit(2)
    print('Validation PASSED')

if __name__ == '__main__':
    main()
