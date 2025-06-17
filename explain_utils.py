import numpy as np
import yaml
from pathlib import Path

LABELS_YAML = Path(__file__).with_suffix(".yml")     # sibling file
with open(LABELS_YAML) as f:
    FRIENDLY_LABELS = yaml.safe_load(f)

def build_reason_strings(shap_vals, feature_names, label_map, n_display=3):
    """
    Returns three arrays:
      • reason_strings (len K)
      • top_feature_labels   (K, n_display)
      • top_shap_values      (K, n_display)
    """
    abs_vals = np.abs(shap_vals)
    order = np.argsort(-abs_vals, axis=1)[:, :n_display]
    top_shap = np.take_along_axis(shap_vals, order, 1)
    top_feat = np.take_along_axis(
        np.asarray(feature_names)[None, :], order, 1
    )

    # Map to business labels and format ±0.00
    labels = np.vectorize(lambda f: label_map.get(f, f))(top_feat)
    contrib = np.round(top_shap, 2)
    reason = [
        "; ".join([f"{l} ({c:+.2f})" for l, c in zip(lbls, conts)])
        for lbls, conts in zip(labels, contrib)
    ]
    return np.array(reason), labels, contrib
