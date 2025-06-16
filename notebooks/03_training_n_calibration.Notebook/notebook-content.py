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
# - Load & Filter: read churn_features, convert call_day, retain last N days for memory
# - Memory Optimizations: downcast dtypes, clip continuous
# - Feature Transforms: log1p, lag/delta, activity flag, replace infinities
# - Train/Val/Test Split (chronological 70/15/15) to prevent leakage
# - Hyperparameter Tuning: RandomizedSearchCV found best params we now use manually
# - Training & Evaluation: fit pipeline, compute AUC on val/test
# - Calibration & Lift Thresholding: sigmoid calibration, custom lift‑optimized classifier, gains/lift charts
# - Persist Artifacts: save training_columns.pkl, churn_pipeline_tuned.joblib to Lakehouse

# CELL ********************

%pip install imbalanced-learn xgboost

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }

# CELL ********************

#== Imports and Config==
import pandas as pd
import numpy as np
import joblib
import logging
from deltalake import DeltaTable
from sklearn.impute import SimpleImputer
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn.calibration import CalibratedClassifierCV
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.base import clone
from builtin.utils.metrics_utils import (
    capacity_pct, precision_at_k, lift_at_k, expected_profit,
    AVG_LOSS_PER_CHURN, CAMPAIGN_COST_PER_CALL)

# Fabric resource imports — shared utilities
from builtin.utils.storage import get_storage_options
from builtin.utils.paths   import PATHS

#= Prevent “Exceeded cell block limit in Agg”=
mpl.rcParams['agg.path.chunksize'] = 10000
#== Define Storage Options and Paths==
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ModelTrain")
storage_options = get_storage_options()
paths           = PATHS

# 1) Load Feature Table
feature_path = paths["churn_features"]
logger.info(f"Loading features from {feature_path}")
df = DeltaTable(feature_path, storage_options=storage_options).to_pandas()
logger.info(f"Loaded df shape: {df.shape}")

# 2) Preprocessing
df["call_day"] = pd.to_datetime(df["call_day"], utc=True, errors="coerce")
max_date = df["call_day"].max()
cutoff = max_date - pd.Timedelta(days=1000)
df = df[df["call_day"] >= cutoff].copy()
logger.info(f"Filtered to last 1000 days: {df.shape}")

# Downcast numeric types
for col in df.select_dtypes(include=["float64"]):
    df[col] = df[col].astype("float32")
for col in df.select_dtypes(include=["int64"]):
    df[col] = df[col].astype("int32")
logger.info("Downcast complete")

# Clip continuous features
clip_cols = [
    "total_calls","total_talk_time","total_amount",
    "rolling_7day_amount","rolling_7day_calls"
]
for c in clip_cols:
    df[c] = df[c].clip(lower=0)

# 3) Hybrid Transformations
df_C = df.copy()

# Log1p transforms
trans_cols = [
    "total_calls","total_talk_time","total_amount",
    "rolling_7day_amount","rolling_7day_calls"
]
for c in trans_cols:
    df_C[f"{c}_trans"] = np.log1p(df_C[c])

# Lag & delta features
df_C = df_C.sort_values(["mid","call_day"])
df_C["lag_total_calls"] = df_C.groupby("mid")["total_calls"].shift(1)
df_C["delta_calls"]     = df_C["total_calls"] - df_C["lag_total_calls"]
if "total_amount" in df_C:
    df_C["lag_total_amount"] = df_C.groupby("mid")["total_amount"].shift(1)
    df_C["delta_amount"]     = df_C["total_amount"] - df_C["lag_total_amount"]

# Activity flag
df_C["activity_flag"] = (
    (df_C["total_calls"] > 0) | (df_C["total_amount"] > 0)
).astype(int)

# Time-based features
df_C["month"]        = df_C["call_day"].dt.month
df_C["day_of_week"]  = df_C["call_day"].dt.dayofweek + 1
df_C["day_of_month"] = df_C["call_day"].dt.day

# 4) Chronological Split
# Build quarterly (≈90-day) hold-out window

df_C = df_C.sort_values("call_day").reset_index(drop=True)

# 4-A  choose cut dates – start after one year of history to guarantee
#      a non-empty training set, stop 90 days before the last record so each fold has a full test window
cut_dates = pd.date_range(
    start=df_C["call_day"].min() + pd.Timedelta(days=365),
    end   =df_C["call_day"].max() - pd.Timedelta(days=90),
    freq  ="90D"          # slide every quarter
)

folds = []       # list of (train_idx, test_idx) boolean masks
for cut in cut_dates:
    tr_idx = df_C["call_day"] < cut
    te_idx = (df_C["call_day"] >= cut) & (df_C["call_day"] < cut + pd.Timedelta(days=90))
    if te_idx.sum() >= 500:        # skip tiny test windows
        folds.append((tr_idx, te_idx))

print(f"Prepared {len(folds)} out-of-time folds "
      f"({cut_dates.freq.n}-day horizon).")


class LiftOptimizedCalibratedClassifier:
    def __init__(self,
                 calibrated_clf,
                 top_percent: float | None = 0.02,
                 fixed_threshold: float | None = None):
        self.calibrated_clf = calibrated_clf
        self.top_percent    = top_percent
        self.threshold      = fixed_threshold if fixed_threshold is not None else 0.5

    def fit(self, X, y):
        if self.top_percent is not None:
            probs = self.calibrated_clf.predict_proba(X)[:, 1]
            dfv   = pd.DataFrame({"label": y, "prob": probs}) \
                     .sort_values("prob", ascending=False) \
                     .reset_index(drop=True)
            cutoff = int(len(dfv) * self.top_percent)
            if cutoff >= 1:
                self.threshold = dfv.loc[cutoff, "prob"]
        return self

    def predict(self, X):
        probs = self.calibrated_clf.predict_proba(X)[:,1]
        return (probs >= self.threshold).astype(int)

    def predict_proba(self, X):
        return self.calibrated_clf.predict_proba(X)



label_col    = "churn_tomorrow"
pos = df_C[label_col].sum()
neg = len(df_C) - pos
scale_pos_weight = neg / pos if pos else 1   # defensive

logger.info(f"scale_pos_weight set to {scale_pos_weight:.1f} "
            f"(neg={neg:,}, pos={pos:,})")

# base pipeline with scale_pos_weight instead of SMOTE
base_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("xgb",     xgb.XGBClassifier(
        use_label_encoder=False,
        eval_metric="auc",
        tree_method="hist",
        random_state=42,
        max_depth=3,
        learning_rate=0.1,
        colsample_bytree=0.5,
        subsample=1.0,
        reg_alpha=0,
        reg_lambda=0.1,
        scale_pos_weight=scale_pos_weight    
    ))
])

logger.info("Pipeline training complete")



# 5) Define Features & Label
base_feats = [
    "total_calls_trans","total_talk_time_trans","total_amount_trans",
    "delta_calls","delta_amount","activity_flag",
    "rolling_7day_amount_trans","rolling_7day_calls_trans",
    "amount_ratio","calls_ratio",
    "days_since_last_tx","days_since_last_activity",
    "calls_ratio_7_30","amount_ratio_7_30",
    "calls_ratio_7_90","amount_ratio_7_90",
    "call_activity_flag","transaction_fee","diff_to_pause_date"
]
pause_cols   = [c for c in df_C.columns if c.startswith("pause_reason_")]
feature_cols = base_feats + pause_cols

CAPACITY_TOP_PCT = capacity_pct(active_count=len(df_C), cadence="weekly")


metrics = []   # to collect AUC per fold


# collect y / p for post-loop plots
calib_data = []  

for i, (tr_idx, te_idx) in enumerate(folds, 1):
    X_tr = df_C.loc[tr_idx, feature_cols]
    y_tr = df_C.loc[tr_idx, label_col]
    X_te = df_C.loc[te_idx, feature_cols]
    y_te = df_C.loc[te_idx, label_col]

    pipe_fold = clone(base_pipeline)
    pipe_fold.fit(X_tr, y_tr)

    calib_fold = CalibratedClassifierCV(pipe_fold, method="sigmoid", cv="prefit")
    calib_fold.fit(X_tr, y_tr)

    lift_clf_fold = LiftOptimizedCalibratedClassifier(
                        calib_fold, top_percent=0.02).fit(X_tr, y_tr)

    proba = lift_clf_fold.predict_proba(X_te)[:, 1]
    auc   = roc_auc_score(y_te, proba)
    prec_k = precision_at_k(y_te, proba, CAPACITY_TOP_PCT)
    lift_k = lift_at_k(y_te, proba, CAPACITY_TOP_PCT)

    # ---------- store metrics ----------
    metrics.append({
        "fold"      : i,
        "train_end" : df_C.loc[tr_idx, "call_day"].max().date(),
        "test_start": df_C.loc[te_idx, "call_day"].min().date(),
        "test_end"  : df_C.loc[te_idx, "call_day"].max().date(),
        "auc"       : round(auc, 4),
        "precision_k": round(prec_k, 3),
        "lift_k"     : round(lift_k, 2)
    })

    # ---------- keep preds for post-run graphics ----------
    calib_data.append({
        "fold": i,
        "y": y_te.values,
        "p": proba
    })

metrics_df = pd.DataFrame(metrics)
display(metrics_df)
print("Mean OOT AUC:", metrics_df["auc"].mean().round(4))

# --- profit-max threshold on last fold 
last_tr_idx, _ = folds[-1]                       # boolean mask
y_val_thr = df_C.loc[last_tr_idx, label_col]
p_val_thr = calib_fold.predict_proba(df_C.loc[last_tr_idx, feature_cols])[:,1]

thr_grid = np.linspace(0.01, 0.50, 50)
profits  = [expected_profit(y_val_thr, p_val_thr, t) for t in thr_grid]
best_thr = thr_grid[int(np.argmax(profits))]
print(f"Best threshold (profit) = {best_thr:.3f}  "
      f"|  €TP={AVG_LOSS_PER_CHURN}, €Call={CAMPAIGN_COST_PER_CALL}")

# 7) Re-train on full history & calibrate 
pipe_full = clone(base_pipeline).fit(df_C[feature_cols], df_C[label_col])

calib_full = CalibratedClassifierCV(pipe_full, method="sigmoid", cv=5)
calib_full.fit(df_C[feature_cols], df_C[label_col])

lift_clf = LiftOptimizedCalibratedClassifier(
               calib_full,
               top_percent=None,
               fixed_threshold=best_thr          # ← use profit threshold
           ).fit(df_C[feature_cols], df_C[label_col])

# 8) Save Artifacts
joblib.dump(feature_cols, paths["training_columns"])
joblib.dump(lift_clf,     paths["model_artifact"])
joblib.dump({"threshold": best_thr,
             "capacity_pct": CAPACITY_TOP_PCT},
            paths["threshold_meta"])
logger.info("Saved training_columns to: %s", paths['training_columns'])
logger.info("Saved calibrated model to: %s", paths['model_artifact'])

# 9) Save raw XGBoost classifier and SHAP explainer for direct use in Notebook 4
# Extract raw classifier from pipeline
xgb_clf = pipe_full.named_steps['xgb']
joblib.dump(xgb_clf, paths.get('xgb_model_unwrapped', '/lakehouse/default/Files/xgb_model_unwrapped.joblib'))
logger.info("Saved unwrapped XGBClassifier to: %s", paths.get('xgb_model_unwrapped'))

# Create and save SHAP TreeExplainer
import shap
explainer = shap.TreeExplainer(xgb_clf)
joblib.dump(explainer, paths.get('shap_explainer', '/lakehouse/default/Files/shap_explainer.joblib'))
logger.info("Saved SHAP TreeExplainer to: %s", paths.get('shap_explainer'))


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }

# CELL ********************

"""%pip install seaborn"""

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python",
# META   "frozen": true,
# META   "editable": false
# META }

# CELL ********************

"""
oot_preds = []
for fdata in calib_data:       # calib_data holds y & p per fold
    y, p, f = fdata["y"], fdata["p"], fdata["fold"]
    idx_te = folds[f-1][1]           # boolean mask for that fold
    dates  = df_C.loc[idx_te, "call_day"].values

    tmp = pd.DataFrame({
        "call_day": dates,
        "y_true":  y,
        "y_prob":  p
    })
    oot_preds.append(tmp)

oot_df = pd.concat(oot_preds, ignore_index=True)
oot_df["month"] = pd.to_datetime(oot_df["call_day"]).dt.to_period("M")
oot_df["y_pred"] = (oot_df["y_prob"] >= best_thr).astype(int)

monthly = (
    oot_df
    .groupby("month")
    .agg(
        actual_churn   = ("y_true", "sum"),
        predicted_flag = ("y_pred", "sum")
    )
    .reset_index()
)"""

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python",
# META   "frozen": true,
# META   "editable": false
# META }

# CELL ********************

"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import PercentFormatter
import matplotlib.patches as mpatches

# Set clean, professional style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Professional color palette
COLORS = {
    'primary': '#2E86AB',
    'secondary': '#A23B72', 
    'success': '#2A9D8F',
    'warning': '#F77F00',
    'danger': '#D62828',
    'neutral': '#6C757D'
}

# ========== 1. LIFT CURVE (YOUR ACTUAL DATA) ==========
def plot_lift_curve(calib_data, capacity_pct=0.02):
    """Plot the actual lift curve from your model"""
    
    # Use last fold data
    last_fold = calib_data[-1]
    y_true = last_fold['y']
    y_prob = last_fold['p']
    
    # Calculate lift curve
    df = pd.DataFrame({'y': y_true, 'prob': y_prob})
    df = df.sort_values('prob', ascending=False).reset_index(drop=True)
    
    df['cum_positive'] = df['y'].cumsum()
    df['cum_percentage'] = (df.index + 1) / len(df) * 100
    df['cum_recall'] = df['cum_positive'] / df['y'].sum()
    df['lift'] = df['cum_recall'] / (df['cum_percentage'] / 100)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot lift curve
    ax.plot(df['cum_percentage'], df['lift'], linewidth=3, color=COLORS['primary'])
    ax.axhline(y=1, color=COLORS['neutral'], linestyle='--', alpha=0.5, label='Random (Lift=1)')
    
    # Highlight top 2%
    idx_2pct = int(len(df) * capacity_pct)
    lift_at_2 = df.iloc[idx_2pct]['lift']
    
    ax.axvline(x=2, color=COLORS['danger'], linestyle=':', linewidth=2, alpha=0.7)
    ax.scatter([2], [lift_at_2], color=COLORS['danger'], s=150, zorder=5)
    ax.annotate(f'Lift = {lift_at_2:.1f}x\nat top 2%', 
                xy=(2, lift_at_2), xytext=(4, lift_at_2 + 3),
                arrowprops=dict(arrowstyle='->', color=COLORS['danger'], lw=1.5),
                fontsize=12, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.3", facecolor='white', edgecolor=COLORS['danger']))
    
    ax.set_xlabel('% of Population (Ranked by Model Score)', fontsize=12)
    ax.set_ylabel('Lift', fontsize=12)
    ax.set_title('Model Lift Curve: Performance at Top 2%', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 10)  # Focus on top 10%
    ax.legend(loc='upper right')
    
    plt.tight_layout()
    plt.show()

# Display the lift curve
plot_lift_curve(calib_data)"""

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python",
# META   "frozen": true,
# META   "editable": false
# META }

# CELL ********************

"""
def plot_cumulative_gains(calib_data, capacity_pct=0.02):
    """Plot cumulative gains showing % of churners captured"""
    
    # Use last fold
    last_fold = calib_data[-1]
    y_true = last_fold['y']
    y_prob = last_fold['p']
    
    # Calculate cumulative gains
    df = pd.DataFrame({'y': y_true, 'prob': y_prob})
    df = df.sort_values('prob', ascending=False).reset_index(drop=True)
    
    total_positives = df['y'].sum()
    df['cum_positives'] = df['y'].cumsum()
    df['pct_total'] = (df.index + 1) / len(df) * 100
    df['pct_positives'] = df['cum_positives'] / total_positives * 100
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot gains curve
    ax.plot(df['pct_total'], df['pct_positives'], linewidth=3, 
            color=COLORS['primary'], label='Model')
    ax.plot([0, 100], [0, 100], '--', color=COLORS['neutral'], 
            alpha=0.5, label='Random')
    
    # Fill area between curves
    ax.fill_between(df['pct_total'], df['pct_positives'], df['pct_total'], 
                    alpha=0.1, color=COLORS['success'])
    
    # Highlight top 2%
    idx_2pct = int(len(df) * capacity_pct)
    gain_at_2 = df.iloc[idx_2pct]['pct_positives']
    
    ax.scatter([2], [gain_at_2], color=COLORS['danger'], s=150, zorder=5)
    ax.plot([2, 2], [0, gain_at_2], ':', color=COLORS['danger'], linewidth=2)
    ax.plot([0, 2], [gain_at_2, gain_at_2], ':', color=COLORS['danger'], linewidth=2)
    
    ax.annotate(f'{gain_at_2:.1f}% of all churners\ncaptured in top 2%', 
                xy=(2, gain_at_2), xytext=(8, gain_at_2 - 10),
                arrowprops=dict(arrowstyle='->', color=COLORS['danger'], lw=1.5),
                fontsize=12, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.3", facecolor='white', edgecolor=COLORS['danger']))
    
    ax.set_xlabel('% of Population Contacted', fontsize=12)
    ax.set_ylabel('% of Churners Captured', fontsize=12)
    ax.set_title('Cumulative Gains: Model Captures Majority of Churners in Top 2%', 
                fontsize=14, fontweight='bold')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 20)  # Focus on top 20%
    ax.set_ylim(0, 100)
    
    plt.tight_layout()
    plt.show()

# Display the lift curve
plot_cumulative_gains(calib_data)
"""

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python",
# META   "frozen": true,
# META   "editable": false
# META }

# CELL ********************

"""def plot_lift_curve_refined(calib_data, capacity_pct=0.02):
    """Professional lift curve with clear value proposition"""
    
    last_fold = calib_data[-1]
    y_true = last_fold['y']
    y_prob = last_fold['p']
    
    df = pd.DataFrame({'y': y_true, 'prob': y_prob})
    df = df.sort_values('prob', ascending=False).reset_index(drop=True)
    
    df['cum_positive'] = df['y'].cumsum()
    df['cum_percentage'] = (df.index + 1) / len(df) * 100
    df['cum_recall'] = df['cum_positive'] / df['y'].sum()
    df['lift'] = df['cum_recall'] / (df['cum_percentage'] / 100)
    
    fig, ax = plt.subplots(figsize=(12, 7.5))
    
    # Main lift curve with gradient
    ax.plot(df['cum_percentage'], df['lift'], linewidth=3.5, color=COLORS['primary'])
    
    # Fill under curve with gradient
    ax.fill_between(df['cum_percentage'], 1, df['lift'], 
                    where=(df['lift'] >= 1), alpha=0.15, color=COLORS['primary'])
    
    # Reference line
    ax.axhline(y=1, color=COLORS['gray'], linestyle='--', alpha=0.6, 
               linewidth=2, label='Random Selection (Lift=1)')
    
    # Highlight operating point
    idx_2pct = int(len(df) * capacity_pct)
    lift_at_2 = df.iloc[idx_2pct]['lift']
    
    ax.axvline(x=2, color=COLORS['danger'], linestyle=':', linewidth=2.5, alpha=0.8)
    ax.scatter([2], [lift_at_2], color=COLORS['danger'], s=200, zorder=5,
               edgecolors='white', linewidth=2)
    
    # Professional annotation
    ax.annotate(f'{lift_at_2:.0f}x better\nthan random', 
                xy=(2, lift_at_2), xytext=(4, lift_at_2 + 5),
                arrowprops=dict(arrowstyle='->', color=COLORS['danger'], lw=2,
                              connectionstyle="arc3,rad=-0.3"),
                fontsize=14, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.5", facecolor='white', 
                         edgecolor=COLORS['danger'], linewidth=2))
    
    # Add efficiency zones
    ax.axhspan(1, 5, alpha=0.05, color=COLORS['gray'], label='Good (1-5x)')
    ax.axhspan(5, 10, alpha=0.05, color=COLORS['olive'], label='Very Good (5-10x)')
    ax.axhspan(10, 50, alpha=0.05, color=COLORS['success'], label='Excellent (>10x)')
    
    # Styling
    ax.set_xlabel('Percentage of Customers Contacted (%)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Lift (Efficiency vs Random Selection)', fontsize=13, fontweight='bold')
    ax.set_title('Model Efficiency: Exceptional Performance in Identifying High-Risk Customers', 
                fontsize=16, fontweight='bold', pad=20)
    
    ax.legend(loc='upper right', frameon=True, fancybox=True, shadow=True, fontsize=11)
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax.set_axisbelow(True)
    
    ax.set_xlim(-0.5, 10)
    ax.set_ylim(0, 35)
    
    # Format x-axis
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, p: f'{x:.0f}%'))
    
    plt.tight_layout()
    return fig

fig = plot_lift_curve_refined(calib_data)
plt.show()"""

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python",
# META   "frozen": true,
# META   "editable": false
# META }

# CELL ********************

"""# ========== 3. OUT-OF-TIME VALIDATION RESULTS ==========
def plot_validation_metrics(metrics_df):
    """Plot the actual validation metrics from your chronological folds"""
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 3.1 AUC over time
    ax = axes[0]
    ax.plot(range(len(metrics_df)), metrics_df['auc'], 
            marker='o', linewidth=2, markersize=8, color=COLORS['primary'])
    ax.axhline(y=metrics_df['auc'].mean(), color=COLORS['secondary'], 
               linestyle='--', alpha=0.7, label=f'Mean: {metrics_df["auc"].mean():.4f}')
    
    ax.set_xlabel('Validation Fold', fontsize=11)
    ax.set_ylabel('AUC Score', fontsize=11)
    ax.set_title('AUC Stability Across Time', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_ylim(metrics_df['auc'].min() - 0.01, metrics_df['auc'].max() + 0.01)
    
    # 3.2 Precision at top 2%
    ax = axes[1]
    ax.plot(range(len(metrics_df)), metrics_df['precision_k'], 
            marker='s', linewidth=2, markersize=8, color=COLORS['success'])
    ax.axhline(y=metrics_df['precision_k'].mean(), color=COLORS['secondary'], 
               linestyle='--', alpha=0.7, label=f'Mean: {metrics_df["precision_k"].mean():.3f}')
    
    ax.set_xlabel('Validation Fold', fontsize=11)
    ax.set_ylabel('Precision at Top 2%', fontsize=11)
    ax.set_title('Precision Stability at Capacity', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    
    # 3.3 Lift at top 2%
    ax = axes[2]
    ax.plot(range(len(metrics_df)), metrics_df['lift_k'], 
            marker='^', linewidth=2, markersize=8, color=COLORS['warning'])
    ax.axhline(y=metrics_df['lift_k'].mean(), color=COLORS['secondary'], 
               linestyle='--', alpha=0.7, label=f'Mean: {metrics_df["lift_k"].mean():.1f}x')
    
    ax.set_xlabel('Validation Fold', fontsize=11)
    ax.set_ylabel('Lift at Top 2%', fontsize=11)
    ax.set_title('Lift Performance Over Time', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.suptitle('Out-of-Time Validation: Consistent Performance Across Quarters', 
                fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.show()

# Display the lift curve
plot_validation_metrics(metrics_df)
"""

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python",
# META   "frozen": true,
# META   "editable": false
# META }

# CELL ********************

"""import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

def plot_score_distribution_refined(calib_data):
    """Professional score distribution with proper scaling, split into two separate figures."""
    
    all_y = np.concatenate([d['y'] for d in calib_data])
    all_p = np.concatenate([d['p'] for d in calib_data])
    
    # —— Figure 1: Log‐scaled histogram of churner vs. non‐churner scores —— #
    churners_scores = all_p[all_y == 1]
    non_churners_scores = all_p[all_y == 0]
    
    threshold_min = 0.0001
    churners_filtered = churners_scores[churners_scores > threshold_min]
    non_churners_filtered = non_churners_scores[non_churners_scores > threshold_min]
    
    bins = np.logspace(np.log10(threshold_min), np.log10(1), 50)
    
    fig1 = plt.figure(figsize=(8, 6))
    ax1 = fig1.add_subplot(1, 1, 1)
    
    ax1.hist(
        non_churners_filtered,
        bins=bins,
        alpha=0.6,
        density=True,
        label=f'Non‐churners (n={len(non_churners_scores):,})',
        color=COLORS['primary'],
        edgecolor='white',
        linewidth=1.2
    )
    ax1.hist(
        churners_filtered,
        bins=bins,
        alpha=0.7,
        density=True,
        label=f'Churners (n={len(churners_scores):,})',
        color=COLORS['danger'],
        edgecolor='white',
        linewidth=1.2
    )
    
    threshold_2pct = np.percentile(all_p, 98)
    ax1.axvline(x=threshold_2pct, color='black', linestyle='--', linewidth=2.5)
    ax1.text(
        threshold_2pct * 1.5,
        ax1.get_ylim()[1] * 0.7,
        'Top 2%\nThreshold',
        fontsize=11,
        fontweight='bold',
        bbox=dict(
            boxstyle="round,pad=0.4",
            facecolor='white',
            edgecolor='black',
            linewidth=1.5
        )
    )
    
    ax1.set_xscale('log')
    ax1.set_xlabel('Model Score (log scale)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Density', fontsize=12, fontweight='bold')
    ax1.set_title(
        'Score Distribution: Churners Concentrate at High Scores',
        fontsize=14,
        fontweight='bold'
    )
    ax1.grid(True, alpha=0.3, which='both')

    return fig1

fig1 = plot_score_distribution_refined(calib_data)
fig1.show()"""

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python",
# META   "frozen": true,
# META   "editable": false
# META }

# CELL ********************

"""import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# --- Compute stability metrics per fold ---------------------------------------
TOP_PCT = 0.02
stability = []

for i, (tr_idx, te_idx) in enumerate(folds, 1):
    y_te    = df_C.loc[te_idx, label_col].values       # 0/1 array
    p_te    = calib_data[i - 1]["p"]                    # numpy array
    thr     = np.quantile(p_te, 1 - TOP_PCT)            # threshold for top 2%

    flagged   = p_te >= thr
    churners  = y_te == 1

    stability.append({
        "test_end"   : df_C.loc[te_idx, "call_day"].max().date(),
        "flag_rate"  : flagged.mean() * 100,                                # %
        "recall_top" : (flagged & churners).sum() / churners.sum() * 100,   # %
        "precision"  : (flagged & churners).sum() / flagged.sum() * 100,    # %
        "lift_top"   : ((flagged & churners).sum() / flagged.sum()) /
                       (churners.mean())                                    # P(churn|flagged)/base
    })

stab_df = pd.DataFrame(stability)
stab_df["test_end"] = pd.to_datetime(stab_df["test_end"])

# --- Time‐Series Dashboard (First‐line only in each panel) --------------------
fig, axes = plt.subplots(
    nrows=2,
    ncols=1,
    figsize=(10, 8),
    sharex=True
)

# PANEL 1 — Only % of pop flagged
axes[0].plot(
    stab_df["test_end"],
    stab_df["flag_rate"],
    marker="s",
    markersize=6,
    linewidth=2,
    label="% of pop flagged",
    color="#d62728"
)
axes[0].set_title("Coverage Over Time", fontsize=14, pad=10)
axes[0].set_ylabel("Percentage", fontsize=12)
axes[0].legend(framealpha=0.7, fontsize=10)
axes[0].grid(alpha=0.3)

# PANEL 2 — Only Precision @ top 2%
axes[1].plot(
    stab_df["test_end"],
    stab_df["precision"],
    marker="D",
    markersize=6,
    linewidth=2,
    label="Precision @ top 2 %",
    color="#2ca02c"
)
axes[1].set_title("Precision Trend", fontsize=14, pad=10)
axes[1].set_ylabel("Value", fontsize=12)
axes[1].set_xlabel("Test Period End", fontsize=12)
axes[1].legend(framealpha=0.7, fontsize=10)
axes[1].grid(alpha=0.3)

# FORMAT X‐AXIS AS DATES
date_fmt = mdates.DateFormatter("%Y-%m")
axes[1].xaxis.set_major_formatter(date_fmt)
for ax in axes:
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=10)

plt.tight_layout()
plt.show()
"""

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python",
# META   "frozen": true,
# META   "editable": false
# META }

# CELL ********************

"""import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# --- Compute stability metrics per fold ---------------------------------------
TOP_PCT = 0.02
stability = []

for i, (tr_idx, te_idx) in enumerate(folds, 1):
    y_te    = df_C.loc[te_idx, label_col].values       # 0/1 array
    p_te    = calib_data[i - 1]["p"]                    # numpy array
    thr     = np.quantile(p_te, 1 - TOP_PCT)            # threshold for top 2%

    flagged   = p_te >= thr
    churners  = y_te == 1

    stability.append({
        "test_end"   : df_C.loc[te_idx, "call_day"].max().date(),
        "recall_top" : (flagged & churners).sum() / churners.sum() * 100,
        "lift_top"   : ((flagged & churners).sum() / flagged.sum()) /
                       (churners.mean())                                    # P(churn|flagged)/base
    })

stab_df = pd.DataFrame(stability)

# Convert 'test_end' into pandas datetime for proper date‐axis formatting
stab_df["test_end"] = pd.to_datetime(stab_df["test_end"])

# --- Simplified Time‐Series Dashboard -----------------------------------------
fig, axes = plt.subplots(
    nrows=2,
    ncols=1,
    figsize=(10, 8),
    sharex=True
)

# PANEL 1 — Recall @ top 2%
axes[0].plot(
    stab_df["test_end"],
    stab_df["recall_top"],
    marker="o",
    markersize=6,
    linewidth=2,
    label="% of churners captured",
    color="#1f77b4"
)
axes[0].set_title("Recall @ Top 2% Over Time", fontsize=14, pad=10)
axes[0].set_ylabel("Percentage", fontsize=12)
axes[0].legend(framealpha=0.7, fontsize=10)
axes[0].grid(alpha=0.3)

# PANEL 2 — Lift @ top 2%
axes[1].plot(
    stab_df["test_end"],
    stab_df["lift_top"],
    marker="^",
    markersize=6,
    linewidth=2,
    label="Lift @ top 2 %",
    color="#9467bd"
)
# Horizontal “excellent” lift benchmark at 10×
axes[1].axhline(
    10,
    linestyle="--",
    linewidth=1,
    color="gray",
    alpha=0.7
)
axes[1].set_title("Lift @ Top 2% Over Time", fontsize=14, pad=10)
axes[1].set_ylabel("Value", fontsize=12)
axes[1].set_xlabel("Test Period End", fontsize=12)
axes[1].legend(framealpha=0.7, fontsize=10)
axes[1].grid(alpha=0.3)

# FORMAT X‐AXIS AS DATES
date_fmt = mdates.DateFormatter("%Y-%m")
axes[1].xaxis.set_major_formatter(date_fmt)
for ax in axes:
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=10)

plt.tight_layout()
plt.show()
"""

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python",
# META   "frozen": true,
# META   "editable": false
# META }

# CELL ********************


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "jupyter_python"
# META }
