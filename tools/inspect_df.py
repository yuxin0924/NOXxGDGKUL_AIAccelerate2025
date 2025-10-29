import os
import sys

# Ensure repo root is on sys.path so we can import Kaixi.feature
repo_root = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, repo_root)

from Kaixi import feature

if __name__ == '__main__':
    # Load and process data using the functions from Kaixi/feature.py
    df_forecast, df_actual, df_dam = feature.load_data()
    df = feature.create_features_and_target(df_forecast, df_actual, df_dam)

    # Print a concise summary
    print('=== Final DataFrame summary from Kaixi.feature.create_features_and_target ===')
    print('shape:', df.shape)
    print('\ncolumns:')
    for c in df.columns:
        print(' -', c)

    print('\n_dtypes:_')
    print(df.dtypes)

    print('\nhead:')
    # Limit printed width
    print(df.head().to_string())

    # Also print index range to show timestamp coverage
    if not df.empty:
        print('\nindex start:', df.index.min())
        print('index end  :', df.index.max())
