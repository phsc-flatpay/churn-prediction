from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import precision_score


# BUSINESS CONSTANTS

# Assumptions:
# Average TPV/month = 17.300€
# margin = Avg. Rate - Trx. cost = 0.56%
# Contr. per customer / month = 97€
# Lifetime = 100 months

AVG_LOSS_PER_CHURN: float     = 9_700.0   # € saved if one merchant is retained
CAMPAIGN_COST_PER_CALL: float =     6.0   # € variable cost per outreach call
WEEKLY_CALL_CAPACITY:  int    =   800     # how many merchants the team can call per week

DEFAULT_CADENCE: str = "weekly"


# DERIVED / HELPER CONSTANTS
# (computed at run-time per scoring batch because active_count can change)
def capacity_pct(active_count: int, cadence: str = DEFAULT_CADENCE) -> float:
    """
    Return the fraction of the portfolio that can be contacted
    """

    if cadence not in ("weekly", "daily"):
        raise ValueError("cadence must be 'weekly' or 'daily'")

    pct = WEEKLY_CALL_CAPACITY / active_count
    return pct if cadence == "weekly" else pct / 5.0   # 5 working days


# METRIC FUNCTIONS
def precision_at_k(y_true: np.ndarray | pd.Series,
                   y_prob: np.ndarray,
                   k_pct: float) -> float:
    """
    Precision among the top-k percent highest-score examples
    """
    k = int(len(y_true) * k_pct)
    if k == 0:
        return np.nan
    idx = np.argsort(y_prob)[::-1][:k]
    return y_true.iloc[idx].mean() if isinstance(y_true, pd.Series) else y_true[idx].mean()


def lift_at_k(y_true: np.ndarray | pd.Series,
              y_prob: np.ndarray,
              k_pct: float) -> float:
    """
    Lift = (precision@k) / (overall positive rate)
    """
    base_rate = y_true.mean()
    return precision_at_k(y_true, y_prob, k_pct) / base_rate if base_rate else np.nan


def expected_profit(y_true: np.ndarray | pd.Series,
                    y_prob: np.ndarray,
                    threshold: float) -> float:
    """
    Expected € profit of contacting everyone above `threshold`
    TP  → value  =  AVG_LOSS_PER_CHURN  (prevented churn)
    FP  → cost   = –CAMPAIGN_COST_PER_CALL
    """

    pred = (y_prob >= threshold).astype(int)
    tp   = ((pred == 1) & (y_true == 1)).sum()
    fp   = ((pred == 1) & (y_true == 0)).sum()
    return tp * AVG_LOSS_PER_CHURN - (tp + fp) * CAMPAIGN_COST_PER_CALL


# what gets exposed from metrics_utils import
__all__ = [
    "AVG_LOSS_PER_CHURN",
    "CAMPAIGN_COST_PER_CALL",
    "WEEKLY_CALL_CAPACITY",
    "capacity_pct",
    "precision_at_k",
    "lift_at_k",
    "expected_profit",
]
