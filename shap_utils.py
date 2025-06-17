import shap
import numpy as np
import pandas as pd
from pathlib import Path
import yaml
from typing import Dict, List, Tuple, Optional


class ShapExplainer:
    """Wrapper for SHAP explainer with feature name mapping"""
    
    def __init__(self, model, feature_cols: List[str], feature_labels_path: Optional[str] = None):
        """
        Initialize SHAP explainer
        
        Args:
            model: The trained XGBoost model (or pipeline)
            feature_cols: List of feature column names
            feature_labels_path: Path to YAML file with human-readable feature labels
        """
        self.feature_cols = feature_cols
        self.feature_labels = self._load_feature_labels(feature_labels_path)
        
        # Extract XGBoost model from pipeline if needed
        if hasattr(model, 'named_steps') and 'xgb' in model.named_steps:
            self.xgb_model = model.named_steps['xgb']
        else:
            self.xgb_model = model
            
        # Create SHAP explainer
        self.explainer = shap.TreeExplainer(self.xgb_model)
        
    def _load_feature_labels(self, path: Optional[str]) -> Dict[str, str]:
        """Load human-readable feature labels from YAML"""
        if path is None:
            # Default path relative to this file
            path = Path(__file__).parent / "explain_utils.yml"
        
        try:
            with open(path, 'r') as f:
                labels = yaml.safe_load(f)
            return labels if labels else {}
        except:
            return {}
    
    def get_feature_label(self, feature_name: str) -> str:
        """Get human-readable label for a feature"""
        # Handle transformed features
        base_feature = feature_name.replace('_trans', '')
        
        if feature_name in self.feature_labels:
            return self.feature_labels[feature_name]
        elif base_feature in self.feature_labels:
            label = self.feature_labels[base_feature]
            if '_trans' in feature_name:
                label += " (log-transformed)"
            return label
        else:
            return feature_name
    
    def explain_prediction(self, X: pd.DataFrame, top_n: int = 3) -> pd.DataFrame:
        """
        Get top contributing features for predictions
        
        Args:
            X: Feature dataframe (can be single row or multiple rows)
            top_n: Number of top features to return
            
        Returns:
            DataFrame with columns: 
            - feature_1, importance_1, ..., feature_n, importance_n
            - combined_explanation
        """
        # Calculate SHAP values
        shap_values = self.explainer.shap_values(X)
        
        # Handle binary classification (take positive class)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        
        # Convert to DataFrame for easier manipulation
        shap_df = pd.DataFrame(shap_values, columns=self.feature_cols)
        
        results = []
        for idx in range(len(X)):
            row_shap = shap_df.iloc[idx]
            row_features = X.iloc[idx]
            
            # Get absolute SHAP values and sort
            abs_shap = row_shap.abs().sort_values(ascending=False)
            
            # Get top features
            top_features = []
            explanations = []
            
            for i in range(min(top_n, len(abs_shap))):
                feature_name = abs_shap.index[i]
                shap_value = row_shap[feature_name]
                feature_value = row_features[feature_name]
                
                # Get human-readable label
                feature_label = self.get_feature_label(feature_name)
                
                # Create explanation
                direction = "increases" if shap_value > 0 else "decreases"
                
                # Format value based on feature type
                if 'ratio' in feature_name or 'flag' in feature_name:
                    value_str = f"{feature_value:.2f}"
                elif 'amount' in feature_name or 'TPV' in feature_name:
                    value_str = f"€{feature_value:,.0f}"
                elif 'days' in feature_name or 'calls' in feature_name:
                    value_str = f"{feature_value:.0f}"
                else:
                    value_str = f"{feature_value:.2f}"
                
                explanation = f"{feature_label} ({value_str}) {direction} risk"
                
                top_features.append({
                    f'feature_{i+1}': feature_label,
                    f'importance_{i+1}': round(shap_value, 4),
                    f'value_{i+1}': feature_value
                })
                
                explanations.append(explanation)
            
            # Combine into single row
            row_result = {}
            for d in top_features:
                row_result.update(d)
            
            # Add combined explanation
            row_result['combined_explanation'] = "; ".join(explanations[:2])  # Top 2 for readability
            
            results.append(row_result)
        
        return pd.DataFrame(results)
    
    def get_feature_importance_summary(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """
        Get overall feature importance across all samples
        
        Returns:
            DataFrame with features ranked by mean absolute SHAP value
        """
        shap_values = self.explainer.shap_values(X)
        
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        
        # Calculate mean absolute SHAP values
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        
        importance_df = pd.DataFrame({
            'feature': self.feature_cols,
            'feature_label': [self.get_feature_label(f) for f in self.feature_cols],
            'importance': mean_abs_shap
        }).sort_values('importance', ascending=False)
        
        return importance_df