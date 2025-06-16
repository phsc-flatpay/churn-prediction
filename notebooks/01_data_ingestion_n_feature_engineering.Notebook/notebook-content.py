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
# - Load & Merge Merchant Master: filter DK customers, select core columns, merge HelloFlow data on mid
# - Clean Phone Numbers: apply clean_phone helper
# - Aggregate Call Logs & Transactions: daily aggregates of calls and talk time; daily sums of transaction amounts
# - Compute Last Activity & Build Skeleton: derive last_activity_day; create merchant–day skeleton respecting activation, deactivation, inactivity
# - Feature Assembly: merge aggregates and master attributes onto skeleton; fill missing; one‑hot encode pause_reason
# - Derived Features: days since activation, pause gap, rolling sums, ratios, calendar flags, holiday indicator
# - Transaction Gap & Activity Flags: compute days since last tx, last activity; clip dates; churn label

# CELL ********************

%pip install pyarrow holidays scikit-learn deltalake duckdb

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }

# CELL ********************

# ---------------------------------------------------------------
# Data Ingestion and Feature Engineering for Churn Prediction
# ---------------------------------------------------------------

import duckdb
import pandas as pd
import numpy as np
import pyarrow as pa
from deltalake import DeltaTable, write_deltalake
from datetime import timedelta
import holidays

from builtin.utils.storage  import get_storage_options
from builtin.utils.paths    import PATHS
from builtin.utils.cleaning import clean_phone

# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------
storage_options = get_storage_options()
paths = PATHS

# Lifetime-TPV filter (EUR)
MIN_LIFETIME_TRX_EUR = 2_500

# ------------------------------------------------------------------
# Helper
# ------------------------------------------------------------------
def to_ns_utc(series: pd.Series) -> pd.Series:
    """Normalize timezone-aware datetime series to datetime64[ns, UTC]"""
    return (
        series
        .dt.tz_convert(None)               # Remove timezone
        .astype("datetime64[ns]")          # Nanosecond precision
        .dt.tz_localize("UTC")             # Apply UTC
    )

# ------------------------------------------------------------------
# Register raw Delta tables in DuckDB
# ------------------------------------------------------------------
con = duckdb.connect()

raw_tables = [
    "fi_db_customer",
    "fi_db_helloflow_customer",
    "fi_db_tx_entries",
    "cc_5749_call_logs",
    "cc_5749_customers",
    "tx_entries"
]

for t in raw_tables:
    ds = DeltaTable(paths[t], storage_options=storage_options).to_pyarrow_dataset()
    con.register(f"{t}_table", ds)

# ------------------------------------------------------------------
# 1)  Build list of eligible mids  (Σ trx_amount_eur > threshold)
# ------------------------------------------------------------------
eligible_mids = (
    con.execute("""
        SELECT  t_mid AS mid
        FROM    tx_entries_table
        WHERE   t_mid IS NOT NULL AND t_mid <> ''
        GROUP BY 1
        HAVING  SUM(trx_amount_eur) > ?
    """, [MIN_LIFETIME_TRX_EUR])    
    .df()["mid"]
    .tolist()
)
print(f"Eligible merchants: {len(eligible_mids)}")

# ------------------------------------------------------------------
# 2)  Load & merge merchant master data  (immediately filter)
# ------------------------------------------------------------------
master_cols = [
    "mid", "friendly_name", "activation_date", "deactived_date",
    "active", "phonenumber", "created",
    "pause_reason", "transaction_fee", "pause_without_usages_list",
    "expected_volume", "acquire",
]

df_master = con.execute(f"""
    SELECT {', '.join(master_cols)}
    FROM   fi_db_customer_table
    WHERE  country = 'DK' AND acquire <> 'RapydEcom'
""").df()

df_master = df_master[df_master["mid"].isin(eligible_mids)]
df_master.drop("acquire", axis=1, inplace=True)

# ------------------------------------------------------------------
# 3)  Helloflow enrichment
# ------------------------------------------------------------------
hf_cols = [
    "mid", "company_name", "friendly_name", "phonenumber",
    "cancel_date", "datetime", "meeting_type_id", "terminals",
]
df_hf = con.execute(f"""
    SELECT {', '.join(hf_cols)}
    FROM   fi_db_helloflow_customer_table
    WHERE  country = 'DK'
""").df()

df_master = df_master.merge(df_hf, on="mid", how="left", suffixes=("", "_hf"))
print("After merchant master merge:", df_master.shape)

# ------------------------------------------------------------------
# 4)  Clean phone numbers
# ------------------------------------------------------------------
df_master["phone_clean"] = df_master["phonenumber"].apply(clean_phone)

# ------------------------------------------------------------------
# 5)  Aggregate call logs (unchanged)
# ------------------------------------------------------------------
df_calls_raw = con.execute("""
    SELECT 
       cl.id           AS call_id,
       cl.created_at,
       cl.talk_time,
       c.phone
    FROM cc_5749_call_logs_table cl
    LEFT JOIN cc_5749_customers_table c
      ON cl.customer_id = c.id
    WHERE c.phone IS NOT NULL
""").df()

df_calls_raw["phone_clean"] = df_calls_raw["phone"].apply(clean_phone)
df_calls_raw["created_at"] = pd.to_datetime(df_calls_raw["created_at"], utc=True, errors="coerce")

df_calls_agg = (
    df_calls_raw
    .groupby(["phone_clean", pd.Grouper(key="created_at", freq="D")])
    .agg(
        total_calls=("call_id", "count"),
        total_talk_time=("talk_time", "sum"),
    )
    .reset_index()
    .rename(columns={"created_at": "call_day"})
)
df_calls_agg["call_day"] = to_ns_utc(df_calls_agg["call_day"])
print("Aggregated calls shape:", df_calls_agg.shape)

# ------------------------------------------------------------------
# 6)  Aggregate transactions (t_mid → mid, trx_amount_eur)
#     and apply the mid filter defensively
# ------------------------------------------------------------------
df_tx = con.execute("""
    SELECT
        mid,
        DATE_TRUNC('day', timestamp) AS call_day,
        SUM(amount) AS total_amount
    FROM fi_db_tx_entries_table          -- <-- correct table for daily feed
    WHERE mid IS NOT NULL
      AND mid <> ''
    GROUP BY mid, DATE_TRUNC('day', timestamp)
""").df()

# keep these two lines exactly as they are
df_tx = df_tx[df_tx["mid"].isin(eligible_mids)]
df_tx["call_day"] = to_ns_utc(pd.to_datetime(df_tx["call_day"], utc=True))

print("Aggregated transactions shape:", df_tx.shape)

# ------------------------------------------------------------------
# 7)  Compute last activity dates
# ------------------------------------------------------------------
df_last_call = (
    df_calls_agg.dropna(subset=["call_day"])
    .groupby("phone_clean", as_index=False)["call_day"]
    .max()
    .rename(columns={"call_day": "last_call_day"})
)

phone_map = df_master[["mid", "phone_clean"]].drop_duplicates()
df_last_call = df_last_call.merge(phone_map, on="phone_clean", how="left")

df_last_tx = (
    df_tx.dropna(subset=["call_day"])
    .groupby("mid", as_index=False)["call_day"]
    .max()
    .rename(columns={"call_day": "last_tx_day"})
)

df_last_activity = pd.merge(df_last_call, df_last_tx, on="mid", how="outer")
df_last_activity["last_activity_day"] = df_last_activity[["last_call_day", "last_tx_day"]].max(axis=1)

df_master = df_master.merge(
    df_last_activity[["mid", "last_activity_day"]],
    on="mid",
    how="left",
)
print("After last_activity merge:", df_master.shape)

# ------------------------------------------------------------------
# 8)  Build merchant-day skeleton (unchanged)
# ------------------------------------------------------------------
df_master["activation_date"] = pd.to_datetime(df_master["activation_date"], utc=True, errors="coerce")
df_master["deactived_date"] = pd.to_datetime(df_master["deactived_date"], utc=True, errors="coerce")

default_date = pd.Timestamp("0001-01-01", tz="UTC")
df_master.loc[df_master["deactived_date"] == default_date, "deactived_date"] = pd.NaT

today = pd.Timestamp.now(tz="UTC")
max_calls_date = df_calls_raw["created_at"].max() if not df_calls_raw.empty else today
max_tx_date = df_tx["call_day"].max() if not df_tx.empty else today
global_max = min(max(max_calls_date, max_tx_date), today)
print("Global maximum date:", global_max)

INACTIVITY_THRESHOLD = 30
rows = []

for _, m in df_master.iterrows():
    if pd.isnull(m["activation_date"]):
        continue

    start_date = m["activation_date"].date()
    end_date = global_max.date()

    if pd.notnull(m["deactived_date"]):
        end_date = min(end_date, (m["deactived_date"] - timedelta(days=1)).date())

    if pd.notnull(m["last_activity_day"]):
        end_date = min(end_date, (m["last_activity_day"] + timedelta(days=INACTIVITY_THRESHOLD)).date())

    if start_date > end_date:
        continue

    rows.extend([(m["mid"], d) for d in pd.date_range(start_date, end_date, freq="D")])

df_skeleton = pd.DataFrame(rows, columns=["mid", "call_day"])
df_skeleton["call_day"] = to_ns_utc(pd.to_datetime(df_skeleton["call_day"], utc=True, errors="coerce"))
print("Skeleton shape:", df_skeleton.shape)

# ------------------------------------------------------------------
# 9)  Merge activity onto skeleton
# ------------------------------------------------------------------
df_skel_calls = pd.merge(
    df_skeleton,
    df_calls_agg,
    left_on=["mid", "call_day"],
    right_on=["phone_clean", "call_day"],
    how="left",
).drop("phone_clean", axis=1)

df_skel_calls["total_calls"] = df_skel_calls["total_calls"].fillna(0)
df_skel_calls["total_talk_time"] = df_skel_calls["total_talk_time"].fillna(0)

df_skel = pd.merge(
    df_skel_calls,
    df_tx,
    on=["mid", "call_day"],
    how="left",
)
df_skel["total_amount"] = df_skel["total_amount"].fillna(0)

# ------------------------------------------------------------------
# 10)  Merge merchant attributes
# ------------------------------------------------------------------
merchant_core_cols = [
    "mid", "activation_date", "deactived_date", "active",
    "pause_reason", "pause_without_usages_list", "transaction_fee",
    "expected_volume", "terminals",
]
merchant_core_cols = [c for c in merchant_core_cols if c in df_master.columns]
df_master_core = df_master[merchant_core_cols].drop_duplicates("mid", keep="first")

df_final = pd.merge(df_skel, df_master_core, on="mid", how="left")
df_final.sort_values("call_day", inplace=True)

# ------------------------------------------------------------------
# 11)  Feature engineering (unchanged logic)
# ------------------------------------------------------------------
df_final["pause_reason"] = df_final["pause_reason"].fillna("no_pause").replace("", "no_pause")
pause_dummies = pd.get_dummies(df_final["pause_reason"], prefix="pause_reason")
df_final = pd.concat([df_final, pause_dummies], axis=1)
df_final.drop(["pause_reason", "pause_reason_no_pause"], axis=1, inplace=True)

df_final["days_since_activation"] = (df_final["call_day"] - df_final["activation_date"]).dt.days

df_final["pause_end_date"] = pd.to_datetime(
    df_final["pause_without_usages_list"], format="%Y-%m-%d", utc=True, errors="coerce"
)
df_final.drop("pause_without_usages_list", axis=1, inplace=True)

valid_year = df_final["pause_end_date"].dt.year.between(2000, 2050)
df_final.loc[~valid_year, "pause_end_date"] = pd.NaT

df_final["diff_to_pause_date"] = np.where(
    df_final["pause_end_date"].notnull(),
    (df_final["call_day"] - df_final["pause_end_date"]).dt.days,
    np.nan,
)

df_final["diff_check"] = (df_final["call_day"] - df_final["pause_end_date"]).dt.days
df_final["diff_mismatch"] = (
    df_final["diff_to_pause_date"].notna()
    & (df_final["diff_to_pause_date"] != df_final["diff_check"])
)
print("diff_mismatch count:", df_final["diff_mismatch"].sum())
df_final.drop(["diff_check", "diff_mismatch"], axis=1, inplace=True)

df_final["pause_end_day"] = df_final["pause_end_date"].dt.day
df_final["pause_end_month"] = df_final["pause_end_date"].dt.month

df_final["transaction_fee"] = df_final["transaction_fee"].clip(lower=0)

for window in [7, 30, 90]:
    df_final[f"rolling_{window}day_amount"] = (
        df_final.groupby("mid")["total_amount"]
        .transform(lambda x: x.rolling(window=window, min_periods=1).sum())
    )
    df_final[f"rolling_{window}day_calls"] = (
        df_final.groupby("mid")["total_calls"]
        .transform(lambda x: x.rolling(window=window, min_periods=1).sum())
    )

df_final["amount_ratio"] = np.where(
    df_final["rolling_7day_amount"] > 0,
    df_final["total_amount"] / df_final["rolling_7day_amount"],
    0,
)
df_final["calls_ratio"] = np.where(
    df_final["rolling_7day_calls"] > 0,
    df_final["total_calls"] / df_final["rolling_7day_calls"],
    0,
)
df_final["amount_ratio_7_30"] = np.where(
    df_final["rolling_30day_amount"] > 0,
    df_final["rolling_7day_amount"] / df_final["rolling_30day_amount"],
    0,
)
df_final["calls_ratio_7_30"] = np.where(
    df_final["rolling_30day_calls"] > 0,
    df_final["rolling_7day_calls"] / df_final["rolling_30day_calls"],
    0,
)
df_final["amount_ratio_7_90"] = np.where(
    df_final["rolling_90day_amount"] > 0,
    df_final["rolling_7day_amount"] / df_final["rolling_90day_amount"],
    0,
)
df_final["calls_ratio_7_90"] = np.where(
    df_final["rolling_90day_calls"] > 0,
    df_final["rolling_7day_calls"] / df_final["rolling_90day_calls"],
    0,
)

df_final["day_of_week"] = df_final["call_day"].dt.dayofweek
df_final["month"] = df_final["call_day"].dt.month
df_final["day_of_month"] = df_final["call_day"].dt.day
df_final["month_cos"] = np.cos(2 * np.pi * df_final["month"] / 12)

try:
    dk_holidays = holidays.Denmark(years=range(2020, 2030))
    df_final["is_holiday"] = df_final["call_day"].dt.date.isin(dk_holidays).astype(int)
except Exception:
    df_final["is_holiday"] = 0

df_tx_sorted = df_tx.sort_values(["mid", "call_day"]).copy()
df_tx_sorted["prev_tx_day"] = df_tx_sorted.groupby("mid")["call_day"].shift(1)
df_tx_sorted["days_since_last_tx"] = (
    df_tx_sorted["call_day"] - df_tx_sorted["prev_tx_day"]
).dt.days

df_final = pd.merge_asof(
    df_final.sort_values("call_day"),
    df_tx_sorted[["mid", "call_day", "days_since_last_tx"]].sort_values("call_day"),
    on="call_day",
    by="mid",
    direction="backward",
)
df_final["days_since_last_tx"] = df_final["days_since_last_tx"].fillna(0)

df_final["call_activity_flag"] = (df_final["total_calls"] > 0).astype(int)
df_final["txn_activity_flag"] = (df_final["total_amount"] > 0).astype(int)

df_activity_dates = (
    df_final[df_final["txn_activity_flag"] == 1][["mid", "call_day"]]
    .drop_duplicates()
    .rename(columns={"call_day": "last_activity_day"})
    .sort_values("last_activity_day")
)

df_final = pd.merge_asof(
    df_final.sort_values("call_day"),
    df_activity_dates,
    left_on="call_day",
    right_on="last_activity_day",
    by="mid",
    direction="backward",
)

df_final["days_since_last_activity"] = (
    df_final["call_day"] - df_final["last_activity_day"]
).dt.days.fillna(0)

activity_counts = (
    df_final.groupby("mid")["txn_activity_flag"].sum().reset_index()
)
activity_counts["no_activity_ever"] = np.where(
    activity_counts["txn_activity_flag"] == 0, 1, 0
)

df_final = pd.merge(
    df_final,
    activity_counts[["mid", "no_activity_ever"]],
    on="mid",
    how="left",
)

min_sql = pd.Timestamp("1753-01-01", tz="UTC")
max_sql = pd.Timestamp("9999-12-31", tz="UTC")
for col in ["activation_date", "deactived_date", "call_day", "pause_end_date"]:
    df_final.loc[df_final[col] < min_sql, col] = pd.NaT
    df_final.loc[df_final[col] > max_sql, col] = pd.NaT

df_final["TPV"] = df_final["expected_volume"] * 12
df_final["TPV_group"] = pd.cut(
    df_final["TPV"],
    bins=[0, 160_000, 650_000, np.inf],
    labels=["low_TPV", "medium_TPV", "high_TPV"],
)

df_final["churn_tomorrow"] = 0
mask_deactivation = (
    df_final["deactived_date"].notna()
    & (df_final["deactived_date"] > df_final["call_day"])
    & ((df_final["deactived_date"] - df_final["call_day"]).dt.days <= 1)
)
df_final.loc[mask_deactivation, "churn_tomorrow"] = 1

mask_no_activity = (
    df_final["days_since_last_activity"].isna()
    & (df_final["no_activity_ever"] == 1)
)
df_final.loc[mask_no_activity, "days_since_last_activity"] = -1
df_final["days_since_last_activity"].fillna(0, inplace=True)

unique_merchants = df_final["mid"].nunique()
churned_merchants = df_final.loc[df_final["churn_tomorrow"] == 1, "mid"].nunique()
print(f"Total merchants: {unique_merchants}")
print(f"Churned merchants: {churned_merchants} ({churned_merchants/unique_merchants:.2%})")

# ------------------------------------------------------------------
# 12)  Write final feature table
# ------------------------------------------------------------------
write_deltalake(
    paths["churn_features"],
    pa.Table.from_pandas(df_final),
    mode="overwrite",
    schema_mode="overwrite",
    storage_options=storage_options,
)
print("Feature table written to:", paths["churn_features"])

# Validate by reading back
df_validation = DeltaTable(
    paths["churn_features"], storage_options=storage_options
).to_pandas()
print("Validation - read back shape:", df_validation.shape)


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }
