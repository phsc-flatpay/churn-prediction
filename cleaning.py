import sys
sys.path.append("Workspaces/Churn/utils")

import pandas as pd

def clean_phone(phone):
    """
    Retains only digits and returns the last 8 digits if length > 8; else returns as-is.
    """
    if pd.isnull(phone):
        return None
    digits = ''.join(filter(str.isdigit, str(phone)))
    return digits[-8:] if len(digits) > 8 else digits
