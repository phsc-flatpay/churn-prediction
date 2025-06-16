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
from builtin.utils.paths import PATHS

# Shared config
storage_options = get_storage_options()
paths = PATHS


def to_ns_utc(series: pd.Series) -> pd.Series:
    """Normalize timezone-aware datetime series to datetime64[ns, UTC]"""
    return (
        series
        .dt.tz_convert(None)               # Remove timezone
        .astype("datetime64[ns]")        # Ensure nanosecond precision
        .dt.tz_localize("UTC")            # Apply UTC timezone
    )

# Register raw tables in DuckDB
con = duckdb.connect()
raw_tables = ["fi_db_customer"]
for table_name in raw_tables:
    dt = DeltaTable(paths[table_name], storage_options=storage_options)
    ds = dt.to_pyarrow_dataset()
    con.register(f"{table_name}_table", ds)

# Load customer data
extra_cols = ["mid", "company_name", "phonenumber", "address", "city", "expected_volume"]
df_customer = con.execute(f"""
    SELECT {', '.join(extra_cols)}
    FROM fi_db_customer_table
    WHERE country = 'DK' AND acquire <> 'RapydEcom'
""").df()

# 1) Re-declare class for unpickling
from sklearn.calibration import CalibratedClassifierCV

class LiftOptimizedCalibratedClassifier:
    def __init__(self, calibrated_clf, top_percent: float | None = 0.02, fixed_threshold: float | None = None):
        self.calibrated_clf = calibrated_clf
        self.top_percent = top_percent
        self.threshold = fixed_threshold if fixed_threshold is not None else 0.5

    def fit(self, X, y):
        # not used in batch scoring
        probs = self.calibrated_clf.predict_proba(X)[:, 1]
        if self.top_percent:
            order = np.argsort(probs)[::-1]
            k = int(len(X) * self.top_percent)
            if k >= 1:
                self.threshold = probs[order[k]]
        return self

    def predict(self, X):
        probs = self.calibrated_clf.predict_proba(X)[:, 1]
        return (probs >= self.threshold).astype(int)

    def predict_proba(self, X):
        return self.calibrated_clf.predict_proba(X)

# 2) Load artifacts
model = joblib.load(paths["model_artifact"])
cols = joblib.load(paths["training_columns"])
meta = joblib.load(paths["threshold_meta"])
thr = meta["threshold"]

# 3) Load features
dt = DeltaTable(paths["churn_features"], storage_options=storage_options)
df = dt.to_pandas()

# 4) Reapply transforms
# 4.1) Filter to last 1000 days
df["call_day"] = pd.to_datetime(df["call_day"], utc=True, errors="coerce")
cutoff_date = df["call_day"].max() - pd.Timedelta(days=1000)
df = df[df["call_day"] >= cutoff_date].copy()

# Additional filters
if "active" in df.columns:
    df = df[df["active"] == 1]
if "diff_to_pause_date" in df.columns:
    df = df[df["diff_to_pause_date"] >= 0]

# 4.2) Downcast numeric types
for c in df.select_dtypes(include=["float64"]):
    df[c] = df[c].astype("float32")
for c in df.select_dtypes(include=["int64"]):
    df[c] = df[c].astype("int32")

# 4.3) Clip continuous features
for c in [
    "total_calls", "total_talk_time", "total_amount",
    "rolling_7day_amount", "rolling_7day_calls", "transaction_fee"
]:
    if c in df.columns:
        df[c] = df[c].clip(lower=0)

# 4.4) Log1p transformations
for c in [
    "total_calls", "total_talk_time", "total_amount",
    "rolling_7day_amount", "rolling_7day_calls"
]:
    if c in df.columns:
        df[f"{c}_trans"] = np.log1p(df[c])

# 4.5) Lag & delta features
df = df.sort_values(["mid", "call_day"])
if "total_calls" in df.columns:
    df["lag_total_calls"] = df.groupby("mid")["total_calls"].shift(1)
    df["delta_calls"] = df["total_calls"] - df["lag_total_calls"]
if "total_amount" in df.columns:
    df["lag_total_amount"] = df.groupby("mid")["total_amount"].shift(1)
    df["delta_amount"] = df["total_amount"] - df["lag_total_amount"]

# 4.6) Activity & time features
df["activity_flag"] = (
    (df.get("total_calls", 0) > 0) | (df.get("total_amount", 0) > 0)
).astype(int)
df["month"] = df["call_day"].dt.month
df["day_of_week"] = df["call_day"].dt.dayofweek + 1
df["day_of_month"] = df["call_day"].dt.day

# 5) Align & Predict
X_new = df.reindex(columns=cols, fill_value=0)
df["churn_probability"] = model.predict_proba(X_new)[:, 1]
df["alert_flag"] = (df["churn_probability"] >= thr).astype(int)

# 6) Latest per Merchant & merge with customer data
df = df.sort_values(["mid", "call_day"])
results = (
    df.groupby("mid", as_index=False)
      .tail(1)[["mid", "churn_probability", "alert_flag"]]
)
results = results.merge(df_customer, on="mid", how="left")

# Compute TPV and TPV_group
results["TPV"] = results["expected_volume"] * 12
results["TPV_group"] = pd.cut(
    results["TPV"], bins=[-np.inf, 160_000, 650_000, np.inf],
    labels=["low_TPV", "medium_TPV", "high_TPV"]
)

# Dynamic cutoff parameters
MAX_MERCHANTS = 400
MIN_PROB_THRESHOLD = 0.005
p_cap = max(0.0, 1.0 - (MAX_MERCHANTS / len(results)))
dyn_thr = results["churn_probability"].quantile(p_cap, interpolation="higher")
cutoff_final = max(dyn_thr, MIN_PROB_THRESHOLD)

# Filter flagged merchants
results = (
    results[results["churn_probability"] >= cutoff_final]
           .sort_values("churn_probability", ascending=False)
           .reset_index(drop=True)
)
results["alert_flag"] = 1
results["cutoff_used"] = cutoff_final

# 7) SHAP explanations
import xgboost as xgb
from sklearn.calibration import CalibratedClassifierCV


def extract_booster(lift_clf: LiftOptimizedCalibratedClassifier) -> xgb.Booster:
    """
    Unwraps the LiftOptimizedCalibratedClassifier to extract the raw XGBoost Booster.
    """
    # 1) calibrated_clf is a CalibratedClassifierCV
    cal = lift_clf.calibrated_clf
    # 2) get a single fitted estimator
    if isinstance(cal, CalibratedClassifierCV):
        # cv=int case -> .calibrated_classifiers_
        candidates = getattr(cal, "calibrated_classifiers_", None)
        if not candidates:
            # cv="prefit" case -> base_estimator
            trained = cal.base_estimator
        else:
            # each candidate is a _CalibratedClassifier wrapper around pipeline
            trained = candidates[0]
    else:
        trained = cal
    # 3) unwrap underlying pipeline if needed
    if hasattr(trained, "base_estimator"):
        trained = trained.base_estimator
    # 4) pipeline -> named_steps
    if hasattr(trained, "named_steps"):
        xgb_clf = trained.named_steps["xgb"]
    else:
        xgb_clf = trained
    # 5) return Booster
    return xgb_clf.get_booster()

booster = extract_booster(model)

# Prepare data for SHAP
result_mids = results["mid"].values

df_shap = (
    df[df["mid"].isin(result_mids)]
      .sort_values(["mid", "call_day"])
      .groupby("mid", as_index=False)
      .tail(1)
)
df_shap = df_shap.set_index("mid").loc[result_mids].reset_index()
X_exp = df_shap[cols].values

# Compute SHAP values
dmat = xgb.DMatrix(X_exp, feature_names=cols)
shap_vals = booster.predict(dmat, pred_contribs=True)[:, :-1]  # drop bias term

# Build human-readable explanations
top_n = 3
explanations = []
for i in range(len(shap_vals)):
    abs_vals = np.abs(shap_vals[i])
    idxs = np.argsort(abs_vals)[-top_n:][::-1]
    reasons = []
    for idx in idxs:
        feat = cols[idx]
        val = shap_vals[i][idx]
        direction = "increases" if val > 0 else "decreases"
        friendly = feat.replace("_trans", "").replace("_", " ").title()
        reasons.append(f"{friendly} {direction} risk")
    explanations.append({
        "mid": result_mids[i],
        "main_reason": reasons[0],
        **{f"driver_{j+1}": cols[idxs[j]] for j in range(top_n)},
        **{f"driver_{j+1}_shap": float(shap_vals[i][idxs[j]]) for j in range(top_n)}
    })

exp_df = pd.DataFrame(explanations)
assert len(exp_df) == len(results), "SHAP explanations count mismatch"
results = results.merge(exp_df, on="mid", how="left")

# 8) Split into segments and persist
for seg in ["low_TPV", "medium_TPV", "high_TPV"]:
    df_out = results[results["TPV_group"] == seg].copy()
    cols_out = [
        "mid", "TPV_group", "company_name", "address", "city", "phonenumber",
        "main_reason",
        *[f"driver_{j}" for j in range(1, top_n+1)],
        *[f"driver_{j}_shap" for j in range(1, top_n+1)]
    ]
    df_out = df_out[cols_out]
    # Ensure no column is all-null
    for c in cols_out:
        if df_out[c].isna().all():
            df_out[c] = df_out[c].astype("string")
    # Set types
    df_out = df_out.astype({
        "mid": "string", "TPV_group": "string",
        "company_name": "string", "address": "string", "city": "string", "phonenumber": "string",
        "main_reason": "string",
        **{f"driver_{j}": "string" for j in range(1, top_n+1)}
    })
    for j in range(1, top_n+1):
        df_out[f"driver_{j}_shap"] = df_out[f"driver_{j}_shap"].astype("float32")
    
    # Local CSV
    csv_name = f"Churn_Predictions_{seg}.csv"
    df_out.to_csv(csv_name, index=False)
    print(f"Saved local CSV: {csv_name} ({len(df_out)} rows)")
    
    # Delta Lake write
    delta_tbl = paths["churn_predictions"].replace(
        "Churn_Predictions", f"Churn_Predictions_{seg}"
    )
    write_deltalake(
        delta_tbl,
        pa.Table.from_pandas(df_out),
        mode="overwrite",
        schema_mode="overwrite",
        storage_options=storage_options
    )
    print(f"Wrote Delta table: {delta_tbl}")


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
from builtin.utils.paths import PATHS

# Shared config
storage_options = get_storage_options()
paths = PATHS


def to_ns_utc(series: pd.Series) -> pd.Series:
    """Normalize timezone-aware datetime series to datetime64[ns, UTC]"""
    return (
        series
        .dt.tz_convert(None)
        .astype("datetime64[ns]")
        .dt.tz_localize("UTC")
    )

# Register raw tables in DuckDB
con = duckdb.connect()
for table_name in ["fi_db_customer"]:
    dt = DeltaTable(paths[table_name], storage_options=storage_options)
    ds = dt.to_pyarrow_dataset()
    con.register(f"{table_name}_table", ds)

# Load customer data
df_customer = con.execute(
    """
    SELECT mid, company_name, phonenumber, address, city, expected_volume
    FROM fi_db_customer_table
    WHERE country='DK' AND acquire<>'RapydEcom'
    """
).df()

# 1) Ensure unpickling works for custom class
from sklearn.calibration import CalibratedClassifierCV

class LiftOptimizedCalibratedClassifier:
    def __init__(self, calibrated_clf, top_percent: float|None=0.02, fixed_threshold: float|None=None):
        self.calibrated_clf = calibrated_clf
        self.top_percent = top_percent
        self.threshold = fixed_threshold if fixed_threshold is not None else 0.5

    def fit(self, X, y):
        probs = self.calibrated_clf.predict_proba(X)[:,1]
        if self.top_percent:
            order = np.argsort(probs)[::-1]
            k = int(len(X)*self.top_percent)
            if k>=1:
                self.threshold = probs[order[k]]
        return self

    def predict(self, X):
        return (self.calibrated_clf.predict_proba(X)[:,1]>=self.threshold).astype(int)

    def predict_proba(self, X):
        return self.calibrated_clf.predict_proba(X)

# 2) Load model & metadata
model: LiftOptimizedCalibratedClassifier = joblib.load(paths["model_artifact"])
cols: list[str] = joblib.load(paths["training_columns"])
meta: dict = joblib.load(paths["threshold_meta"])
thr: float = meta.get("threshold", 0.5)

# 3) Load feature table
dt = DeltaTable(paths["churn_features"], storage_options=storage_options)
df = dt.to_pandas()

# 4) Preprocess feature table
# 4.1) Date filter: last 1000d
df["call_day"] = pd.to_datetime(df["call_day"], utc=True, errors="coerce")
c_date = df["call_day"].max() - pd.Timedelta(days=1000)
df = df[df["call_day"]>=c_date].copy()
# 4.2) Additional filters
for col, cond in [("active", df["active"]==1 if "active" in df else None),
                  ("diff_to_pause_date", df["diff_to_pause_date"]>=0 if "diff_to_pause_date" in df else None)]:
    if cond is not None:
        df = df[cond].copy()
# 4.3) Downcast
dtypes = {"float64":"float32","int64":"int32"}
for dt_in, dt_out in dtypes.items():
    for c in df.select_dtypes(include=[dt_in]): df[c]=df[c].astype(dt_out)
# 4.4) Clip & log1p
for c in ["total_calls","total_talk_time","total_amount","rolling_7day_amount","rolling_7day_calls","transaction_fee"]:
    if c in df: df[c]=df[c].clip(0)
for c in ["total_calls","total_talk_time","total_amount","rolling_7day_amount","rolling_7day_calls"]:
    if c in df: df[f"{c}_trans"] = np.log1p(df[c])
# 4.5) Lag & delta
df = df.sort_values(["mid","call_day"])
if "total_calls" in df:
    df["lag_total_calls"] = df.groupby("mid")["total_calls"].shift(1)
    df["delta_calls"] = df["total_calls"]-df["lag_total_calls"]
if "total_amount" in df:
    df["lag_total_amount"] = df.groupby("mid")["total_amount"].shift(1)
    df["delta_amount"] = df["total_amount"]-df["lag_total_amount"]
# 4.6) Activity & time
df["activity_flag"] = ((df.get("total_calls",0)>0)|(df.get("total_amount",0)>0)).astype(int)
df["month"]=df["call_day"].dt.month
df["day_of_week"]=df["call_day"].dt.dayofweek+1
df["day_of_month"]=df["call_day"].dt.day

# 5) Generate predictions
X_new = df.reindex(columns=cols, fill_value=0)
df["churn_probability"]=model.predict_proba(X_new)[:,1]
df["alert_flag"]=(df["churn_probability"]>=thr).astype(int)

# 6) Build results table
df_latest = df.sort_values(["mid","call_day"]).groupby("mid",as_index=False).tail(1)
results = df_latest[["mid","churn_probability","alert_flag"]].merge(df_customer,on="mid",how="left")
# TPV grouping
results["TPV"]=results["expected_volume"]*12
results["TPV_group"]=pd.cut(results["TPV"],bins=[-np.inf,160_000,650_000,np.inf],labels=["low_TPV","medium_TPV","high_TPV"])
# Dynamic cutoff
MAX=400; MIN=0.005
pc=max(0.0,1- MAX/len(results))
th=results["churn_probability"].quantile(pc,interpolation="higher")
cut=max(th,MIN)
results=results[results["churn_probability"]>=cut].sort_values("churn_probability",ascending=False).reset_index(drop=True)
results["alert_flag"]=1;results["cutoff_used"]=cut

# 7) SHAP explanations using shap.Explainer on the XGBClassifier
import shap
from sklearn.calibration import CalibratedClassifierCV

# 7.1) Extract the raw XGBClassifier from the calibrated wrapper
cal_clf = model.calibrated_clf
if isinstance(cal_clf, CalibratedClassifierCV) and hasattr(cal_clf, 'calibrated_classifiers_'):
    wrapper = cal_clf.calibrated_classifiers_[0]
    base_est = getattr(wrapper, 'estimator_', wrapper)
elif hasattr(cal_clf, 'base_estimator'):
    base_est = cal_clf.base_estimator
else:
    base_est = cal_clf
if hasattr(base_est, 'named_steps') and 'xgb' in base_est.named_steps:
    xgb_clf = base_est.named_steps['xgb']
else:
    xgb_clf = base_est

# 7.2) Prepare data for explanations
mids = results['mid'].values

df_sh = (
    df[df['mid'].isin(mids)]
      .sort_values(['mid','call_day'])
      .groupby('mid', as_index=False)
      .tail(1)
)
df_sh = df_sh.set_index('mid').loc[mids].reset_index()
Xexp = df_sh[cols]

# 7.3) Compute SHAP values with shap.Explainer on XGBClassifier
explainer = shap.Explainer(xgb_clf, Xexp)
# Depending on SHAP version, use .shap_values or call
try:
    shap_matrix = explainer.shap_values(Xexp)
except AttributeError:
    shap_matrix = explainer(Xexp).values

# 7.4) Build explanations list
top_n = 3
explanations = []
for i in range(shap_matrix.shape[0]):
    row = shap_matrix[i]
    idxs = np.argsort(np.abs(row))[-top_n:][::-1]
    reasons = []
    for idx in idxs:
        feat = cols[idx]
        val = row[idx]
        direction = 'increases' if val > 0 else 'decreases'
        friendly = feat.replace('_trans','').replace('_',' ').title()
        reasons.append(f"{friendly} {direction} risk")
    entry = {'mid': mids[i], 'main_reason': reasons[0]}
    for j, idx in enumerate(idxs, start=1):
        entry[f'driver_{j}'] = cols[idx]
        entry[f'driver_{j}_shap'] = float(row[idx])
    explanations.append(entry)
exp_df = pd.DataFrame(explanations)
assert exp_df.shape[0] == results.shape[0], 'SHAP explanations count mismatch'
# Merge explanations into results
results = results.merge(exp_df, on='mid', how='left')

# 8) Persist per segment
for seg in ["low_TPV","medium_TPV","high_TPV"]:
    df_out=results[results.TPV_group==seg].copy()
    out_cols=["mid","TPV_group","company_name","address","city","phonenumber","main_reason"]
    for i in range(1,top_n+1): out_cols.append(f"driver_{i}"),out_cols.append(f"driver_{i}_shap")
    df_out=df_out[out_cols]
    # ensure types
    for c in out_cols:
        if df_out[c].isna().all(): df_out[c]=df_out[c].astype("string")
    df_out=df_out.astype({col:("string" if not col.endswith("_shap") else "float32") for col in df_out.columns})
    # write CSV
    fn=f"Churn_Predictions_{seg}.csv";df_out.to_csv(fn,index=False);print(f"Saved {fn} ({len(df_out)}) rows")
    # write Delta
    tbl=paths["churn_predictions"].replace("Churn_Predictions",f"Churn_Predictions_{seg}")
    write_deltalake(tbl,pa.Table.from_pandas(df_out),mode="overwrite",schema_mode="overwrite",storage_options=storage_options)
    print(f"Wrote {tbl}")


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }

# CELL ********************

# 7) Persist each list
for seg, df_seg in lists.items():
    # Select final columns INCLUDING EXPLANATIONS
    df_seg = df_seg[
        ["mid", "TPV_group", "company_name", "address", "city", "phonenumber",
         "main_reason", "driver_1", "driver_1_shap", "driver_2", "driver_2_shap", 
         "driver_3", "driver_3_shap"]
    ].copy()
    
    # Ensure no columns have Null type
    for col in df_seg.columns:
        if df_seg[col].isna().all():
            df_seg[col] = df_seg[col].astype('string')
    
    # Set proper types
    df_seg = df_seg.astype({
        'mid': 'string',
        'TPV_group': 'string',
        'company_name': 'string',
        'address': 'string',
        'city': 'string',
        'phonenumber': 'string',
        'main_reason': 'string',
        'driver_1': 'string',
        'driver_2': 'string',
        'driver_3': 'string'
    })
    
    # Keep SHAP values as float
    df_seg['driver_1_shap'] = df_seg['driver_1_shap'].astype('float32')
    df_seg['driver_2_shap'] = df_seg['driver_2_shap'].astype('float32')
    df_seg['driver_3_shap'] = df_seg['driver_3_shap'].astype('float32')
    
    csv_name = f"Churn_Predictions_{seg}.csv"
    delta_tbl = paths["churn_predictions"].replace(
        "Churn_Predictions", f"Churn_Predictions_{seg}"
    )

    # 7-A) local CSV
    df_seg.to_csv(csv_name, index=False)
    print(f"Local CSV saved: {csv_name}  ({len(df_seg):,} rows)")

    # 7-B) Delta table
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

# 7) Persist each list
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

    # 7-A) local CSV
    df_seg.to_csv(csv_name, index=False)
    print(f"Local CSV saved: {csv_name}  ({len(df_seg):,} rows)")

    # 7-B) Delta table
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

"""import numpy as np
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
quantile_grid = np.arange(0.90, 1.0001, 0.005)        # 90 % … 100 % in 0.5 % steps
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
"""

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }

# CELL ********************

# --------------------------------------------------------------
# 8)  Row‑level explanations (top‑K by predicted probability)
# --------------------------------------------------------------
TOPK_FOR_EXPLANATIONS = 2_000        # set 0 / None to disable

from builtin.utils.explain_utils import build_reason_strings, FRIENDLY_LABELS
import xgboost as xgb, numpy as np, pyarrow as pa
from deltalake import write_deltalake

# --- 8.1  Extract the raw Booster from the wrapped model ----------
def _extract_booster(lift_clf):
    cal = lift_clf.calibrated_clf
    inner = cal.base_estimator if getattr(cal, "base_estimator", None) else cal.calibrated_classifiers_[0]
    pipe  = inner.base_estimator if getattr(inner, "base_estimator", None) else inner.estimator
    xgb_clf = pipe.named_steps["xgb"] if hasattr(pipe, "named_steps") else pipe
    return xgb_clf.get_booster()

booster = _extract_booster(model)

# --- 8.2  Pick global top‑K rows ----------------------------------
y_pred  = df["churn_probability"].to_numpy()
K       = min(TOPK_FOR_EXPLANATIONS, len(df))
top_idx = np.argsort(-y_pred)[:K]

feature_names = getattr(model.calibrated_clf, "feature_names_in_", None)
if feature_names is None or len(feature_names) == 0:
    feature_names = joblib.load(paths["training_columns"])

dmat_top  = xgb.DMatrix(X_new.iloc[top_idx].values, feature_names=feature_names)
shap_vals = booster.predict(dmat_top, pred_contribs=True)[:, :-1]  # drop bias term

# --- 8.3  Build human‑readable reason strings ---------------------
reasons, top_feats, top_contribs = build_reason_strings(
    shap_vals, feature_names, FRIENDLY_LABELS, n_display=2
)

# --- 8.4  Attach to the outbound 200‑row results ------------------
map_mid = {
    df.iloc[i]["mid"]: {
        "reason": reasons[j],
        "driver_1": top_feats[j, 0], "driver_1_shap": float(top_contribs[j, 0]),
        "driver_2": top_feats[j, 1], "driver_2_shap": float(top_contribs[j, 1]),
    } for j, i in enumerate(top_idx)
}

expl_df = results["mid"].map(map_mid).apply(pd.Series)
for c in expl_df.columns:                      # overwrite / create safely
    results[c] = expl_df[c]

# --- 8.5  Persist explanations to Delta (pure Python) -------------
delta_path = paths["churn_predictions"].replace(
    "Churn_Predictions", "Churn_Reasons_Daily"
)
cols_save  = ["mid", "churn_probability", "TPV_group",
              "reason", "driver_1", "driver_1_shap",
              "driver_2", "driver_2_shap"]

write_deltalake(
    delta_path,
    pa.Table.from_pandas(results[cols_save]),
    mode="append",
    schema_mode="merge",
    storage_options=storage_options
)

print(f"Row‑level explanations written for {len(results)} merchants → {delta_path}")


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
