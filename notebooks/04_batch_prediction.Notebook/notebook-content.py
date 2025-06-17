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
# - Load Artifacts & Features: training_columns, calibrated model, latest churn_features
# - Filter Active Merchants & Transform: filter active==1, reapply date filter, downcast, clip, log1p, lag/delta
# - Align Columns: reindex to training_columns
# - Predict: compute probabilities, pick most recent record per mid
# - Write Predictions: save CSV or Delta table to Lakehouse

# CELL ********************

%pip install imbalanced-learn

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }

# CELL ********************


import pandas as pd
import numpy as np
import joblib
import pyarrow as pa
from deltalake import DeltaTable, write_deltalake
import duckdb

# Fabric resource imports — shared utilities
from builtin.utils.storage import get_storage_options
from builtin.utils.paths   import PATHS

# Shared config
storage_options = get_storage_options()
paths           = PATHS


def to_ns_utc(series: pd.Series) -> pd.Series:
    """Normalize timezone-aware datetime series to datetime64[ns, UTC]"""
    return (
        series
        .dt.tz_convert(None)               # Remove timezone
        .astype("datetime64[ns]")          # Ensure nanosecond precision
        .dt.tz_localize("UTC")             # Apply UTC timezone
    )

# Register raw tables in DuckDB
con = duckdb.connect()
raw_tables = [
    "fi_db_customer"]

for table_name in raw_tables:
    dt = DeltaTable(paths[table_name], storage_options=storage_options)
    ds = dt.to_pyarrow_dataset()
    con.register(f"{table_name}_table", ds)

# Load customer data with required columns
extra_cols = ["mid", "company_name", "phonenumber", "address", "city", "expected_volume"]

df_customer = con.execute(f"""
    SELECT {', '.join(extra_cols)}
    FROM fi_db_customer_table
    WHERE country = 'DK' AND acquire <> 'RapydEcom'
""").df()


# Re-declare the custom class so unpickling works
from sklearn.calibration import CalibratedClassifierCV

class LiftOptimizedCalibratedClassifier:
    def __init__(self,
                 calibrated_clf,
                 top_percent: float | None = 0.02,
                 fixed_threshold: float | None = None):
        self.calibrated_clf = calibrated_clf
        self.top_percent    = top_percent
        # value was persisted during training; we do not modify it here
        self.threshold      = fixed_threshold if fixed_threshold is not None else 0.5

    def fit(self, X, y):
        # not used in batch scoring, but kept for completeness
        probs  = self.calibrated_clf.predict_proba(X)[:, 1]
        order  = np.argsort(probs)[::-1]
        if self.top_percent:
            k = int(len(X) * self.top_percent)
            if k >= 1:
                self.threshold = probs[order[k]]
        return self

    def predict(self, X):
        probs = self.calibrated_clf.predict_proba(X)[:, 1]
        return (probs >= self.threshold).astype(int)

    def predict_proba(self, X):
        return self.calibrated_clf.predict_proba(X)

# Load Model & Feature Columns
model = joblib.load(paths["model_artifact"])
cols  = joblib.load(paths["training_columns"])
meta = joblib.load(paths["threshold_meta"]) # from nb 3
thr  = meta["threshold"]

# Load Latest Features
dt = DeltaTable(paths["churn_features"], storage_options=storage_options)
df = dt.to_pandas()

# Reapply Transforms (last 1000 days, downcast, clip, log1p, lag/delta, activity, time)
# Filter to recent
df["call_day"] = pd.to_datetime(df["call_day"], utc=True, errors="coerce")
cutoff = df["call_day"].max() - pd.Timedelta(days=1000)
df = df[df["call_day"] >= cutoff].copy()

# filter for active
if "active" in df.columns:
    df = df[df["active"] == 1].copy()

# Downcast numeric types
for c in df.select_dtypes(include=["float64"]):
    df[c] = df[c].astype("float32")
for c in df.select_dtypes(include=["int64"]):
    df[c] = df[c].astype("int32")

# Clip continuous
for c in [
    "total_calls","total_talk_time","total_amount",
    "rolling_7day_amount","rolling_7day_calls","transaction_fee"
]:
    df[c] = df[c].clip(lower=0)

# Log1p transforms
for c in [
    "total_calls","total_talk_time","total_amount",
    "rolling_7day_amount","rolling_7day_calls"
]:
    df[f"{c}_trans"] = np.log1p(df[c])

# Lag & delta
df = df.sort_values(["mid","call_day"])
df["lag_total_calls"] = df.groupby("mid")["total_calls"].shift(1)
df["delta_calls"]     = df["total_calls"] - df["lag_total_calls"]
if "total_amount" in df:
    df["lag_total_amount"] = df.groupby("mid")["total_amount"].shift(1)
    df["delta_amount"]     = df["total_amount"] - df["lag_total_amount"]

# Activity & time features
df["activity_flag"]   = ((df["total_calls"] > 0) | (df["total_amount"] > 0)).astype(int)
df["month"]           = df["call_day"].dt.month
df["day_of_week"]     = df["call_day"].dt.dayofweek + 1
df["day_of_month"]    = df["call_day"].dt.day

# align & Predict
X_new = df[cols].reindex(columns=cols, fill_value=0)
df["churn_probability"] = model.predict_proba(X_new)[:,1]

df["alert_flag"] = (df["churn_probability"] >= thr).astype(int)
print(f"{df['alert_flag'].sum()} merchants flagged (prob ≥ {thr:.3f})")


# Latest per Merchant & Annotate TPV Group
# Get most-recent snapshot per merchant
latest = (
    df.sort_values(["mid", "call_day"])
      .groupby("mid", as_index=False)
      .tail(1)
)
latest = latest[latest["diff_to_pause_date"] >= 0].copy()

#Narrow to prediction columns for downstream code
results = latest[["mid", "churn_probability", "alert_flag", "expected_volume"]].copy()

if "expected_volume" in results.columns:
    results = results.drop(columns=["expected_volume"])

results = results.merge(df_customer, on="mid", how="left")

# Compute TPV group
results["TPV"] = results["expected_volume"] * 12
results["TPV_group"] = pd.cut(
    results["TPV"],
    bins=[-np.inf, 160_000, 650_000, np.inf],
    labels=["low_TPV", "medium_TPV", "high_TPV"]
)


# DYNAMIC CUTOFF
MAX_MERCHANTS     = 400          # max analyst capacity
MIN_PROB_THRESHOLD = 0.003           # never alert below x%, larger -> less

# dynamic percentile to respect capacity
p_cap   = max(0.0, 1.0 - (MAX_MERCHANTS / len(results)))
dyn_thr = results["churn_probability"].quantile(
              p_cap, interpolation="higher")

cutoff  = max(dyn_thr, MIN_PROB_THRESHOLD)
print(f"{len(results)} merchants flagged "
      f"(prob ≥ {cutoff:.4%}, dyn={dyn_thr:.4%}, floor={MIN_PROB_THRESHOLD:.4%})")



# recompute alert flag on the final set
results = (
    results
    .loc[results["churn_probability"] >= cutoff]
    .sort_values("churn_probability", ascending=False)
    .reset_index(drop=True)
)
results["alert_flag"]  = 1
results["cutoff_used"] = cutoff


# Split into three segment-specific DataFrames
lists = {
    "low_TPV":    results[results["TPV_group"] == "low_TPV"].copy(),
    "medium_TPV": results[results["TPV_group"] == "medium_TPV"].copy(),
    "high_TPV":   results[results["TPV_group"] == "high_TPV"].copy()
}

# Persist each list
for seg, df_seg in lists.items():
    # Select final columns
    df_seg = df_seg[
        ["mid", "TPV_group", "company_name", "address", "city", "phonenumber"]
    ].copy()
    
    # Ensure no columns have Null type by converting to appropriate string types
    for col in df_seg.columns:
        if df_seg[col].isna().all():  # If column is entirely null
            df_seg[col] = df_seg[col].astype('string')  # Convert to string type
    
    # Also ensure other columns have proper types
    df_seg = df_seg.astype({
        'mid': 'string',
        'TPV_group': 'string',
        'company_name': 'string',
        'address': 'string',
        'city': 'string',
        'phonenumber': 'string'
    })
    
    csv_name = f"Churn_Predictions_{seg}.csv"
    delta_tbl = paths["churn_predictions"].replace(
        "Churn_Predictions", f"Churn_Predictions_{seg}"
    )

    # local CSV
    df_seg.to_csv(csv_name, index=False)
    print(f"Local CSV saved: {csv_name}  ({len(df_seg):,} rows)")

    # Delta table
    write_deltalake(
        delta_tbl,
        pa.Table.from_pandas(df_seg),
        mode="overwrite",
        schema_mode="overwrite",
        storage_options=storage_options
    )
    print(f"{seg} list written to: {delta_tbl}")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }

# CELL ********************

import numpy as np
import pandas as pd

probs = results["churn_probability"]
n     = len(probs)

# ────────────────────────────────────────────────────────────────────
# 1. Descriptive stats
# ────────────────────────────────────────────────────────────────────
print("\n*** Summary statistics ***")
print(probs.describe(percentiles=[.90,.95,.975,.98,.99,.995]).to_string(float_format="%.6f"))

# ────────────────────────────────────────────────────────────────────
# 2. Counts above common percentile thresholds
# ────────────────────────────────────────────────────────────────────
quantile_grid = np.arange(0.90, 1.0001, 0.005)        # 90-100 % in 0.5 % steps
thresh        = probs.quantile(quantile_grid)

summary = (
    pd.DataFrame({
        "quantile"        : quantile_grid,
        "score_threshold" : thresh.values,
        "count_above"     : [(probs >= t).sum() for t in thresh.values]
    })
    .assign(pct_of_total = lambda df_: (df_["count_above"] / n).round(4))
)

print("\n*** Merchants above each percentile threshold ***")
print(summary.to_string(index=False, float_format="%.6f"))

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
