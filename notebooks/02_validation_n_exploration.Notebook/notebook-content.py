# Fabric notebook source

# METADATA ********************

# META {
# META   "kernel_info": {
# META     "name": "jupyter",
# META     "jupyter_kernel_name": "python3.11"
# META   },
# META   "dependencies": {
# META     "lakehouse": {
# META       "default_lakehouse": "54c8eb17-9ab5-4b14-9a01-728630a243fb",
# META       "default_lakehouse_name": "ml_ops",
# META       "default_lakehouse_workspace_id": "e03dbe51-4f30-4e31-a84f-647f6b831f58",
# META       "known_lakehouses": [
# META         {
# META           "id": "54c8eb17-9ab5-4b14-9a01-728630a243fb"
# META         },
# META         {
# META           "id": "975b4954-52d5-4adb-baab-c6cd6452c8e5"
# META         },
# META         {
# META           "id": "f51e4b7c-79a8-47c3-91b7-5cbd92bf65b0"
# META         }
# META       ]
# META     }
# META   }
# META }

# MARKDOWN ********************

# **Outline:**
# - Load Feature Table
# - Sanity Tests:
#     - Schema consistency
#     - Date ranges & logical checks
#     - Missing values in critical fields
#     - Cross-field consistency (e.g. diff_to_pause_date)
#     - Numeric distributions & ratio bounds
# - Correlation Analysis: heatmap & top features vs. churn_tomorrow

# CELL ********************

%pip install pyarrow holidays scikit-learn deltalake duckdb matplotlib seaborn

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }

# CELL ********************

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from deltalake import DeltaTable

from builtin.utils.storage import get_storage_options
from builtin.utils.paths import PATHS
from builtin.utils.cleaning import clean_phone

# Configuration
storage_options = get_storage_options()
paths = PATHS

# Load churn features data
dt = DeltaTable(paths["churn_features"], storage_options=storage_options)
df = dt.to_pandas()

# Calculate business capacity baseline
from builtin.utils.metrics_utils import capacity_pct, precision_at_k, lift_at_k, expected_profit

CAPACITY_TOP_PCT = capacity_pct(active_count=len(df), cadence="weekly")
print(f"Weekly call capacity = {CAPACITY_TOP_PCT*100:.2f}% of the portfolio "
      f"({len(df)*CAPACITY_TOP_PCT:.0f} merchants).")


# Schema Consistency Test
def test_schema_consistency(df, expected):
    """Verify that DataFrame columns match expected schema."""
    print("Schema Consistency")
    print("-" * 50)
    
    for col, exp in expected.items():
        act = str(df[col].dtype) if col in df else "MISSING"
        print(f"{col:30s}: {act:20s} (expected {exp})")
    
    print("\nFull dtypes:")
    print(df.dtypes)


expected_columns = {
    'mid': 'object',
    'call_day': 'datetime64[ns, UTC]',
    'total_calls': 'float64',
    'total_talk_time': 'float64',
    'total_amount': 'float64',
    'activation_date': 'datetime64[ns, UTC]',
    'deactived_date': 'datetime64[ns, UTC]',
    'active': 'int32',
    'transaction_fee': 'float64',
    'churn_tomorrow': 'int64'
}

test_schema_consistency(df, expected_columns)


# Date Columns Verification
def test_date_columns(df, date_cols=None):
    """Check date column ranges and validity."""
    print("\nDate Columns")
    print("-" * 50)
    
    if date_cols is None:
        date_cols = ['call_day', 'activation_date', 'deactived_date', 'pause_end_date']
    
    for col in date_cols:
        if col in df:
            non_null = df[col].dropna()
            if not non_null.empty:
                print(f"{col}: min={non_null.min()}, max={non_null.max()}")
            else:
                print(f"{col}: all NaT")
        else:
            print(f"{col}: NOT FOUND")


test_date_columns(df)


# Date Ranges and Logical Consistency
print("\nDate Ranges & Logic")
print("-" * 50)

now = pd.Timestamp.now(tz='UTC')
print(f"call_day > now: {(df['call_day'] > now).sum()}")
print(f"activation_date > call_day: {((df['activation_date'] > df['call_day']) & df['activation_date'].notna()).sum()}")
print(f"deactived_date < activation_date: {((df['deactived_date'] < df['activation_date']) & df['deactived_date'].notna() & df['activation_date'].notna()).sum()}")


# Missing Values in Critical Fields
print("\nMissing Values")
print("-" * 50)

for col in ['mid', 'call_day', 'activation_date']:
    pct = df[col].isna().mean() * 100
    print(f"{col}: {pct:.2f}% missing")


# Cross-Field Consistency
print("\nCross-Field Consistency")
print("-" * 50)

mismatches = df.loc[
    (df['diff_to_pause_date'].notna()) &
    (df['diff_to_pause_date'] != (df['call_day'] - df['pause_end_date']).dt.days)
]
print(f"diff_to_pause_date mismatches: {mismatches.shape[0]}")


# Numeric and Ratio Checks
print("\nNumeric & Ratio Stats")
print("-" * 50)

ratio_cols = [
    'amount_ratio', 'calls_ratio',
    'amount_ratio_7_30', 'calls_ratio_7_30',
    'amount_ratio_7_90', 'calls_ratio_7_90'
]

for c in ratio_cols:
    if c in df:
        print(f"{c}: min={df[c].min():.3f}, max={df[c].max():.3f}, mean={df[c].mean():.3f}")


# Statistical Summary of Key Numerics
print("\nNumeric Summary")
print("-" * 50)

key_numeric_cols = ['total_calls', 'total_talk_time', 'total_amount', 'transaction_fee']
print(df[key_numeric_cols].describe())


# Correlation Analysis
numeric_df = df.select_dtypes(include=['float64', 'int64'])
corr = numeric_df.corr()

# Plot correlation heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(corr, cmap='coolwarm', annot=False)
plt.title("Feature Correlation Matrix")
plt.tight_layout()
plt.show()

# Top features correlated with churn
churn_corr = corr['churn_tomorrow'].abs().sort_values(ascending=False)
print("\nTop features correlated with churn_tomorrow:")
print(churn_corr.head(10))

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }

# CELL ********************


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }
