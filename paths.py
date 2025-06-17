# paths.py

PATHS = {
    # raw input tables (unchanged)…
    "fi_db_customer":
        "abfss://4ee0ce3e-b5a2-474a-8a0f-1fc7484a94fc"
        "@onelake.dfs.fabric.microsoft.com/"
        "975b4954-52d5-4adb-baab-c6cd6452c8e5/Tables/fi_db_customer",

    "fi_db_helloflow_customer":
        "abfss://4ee0ce3e-b5a2-474a-8a0f-1fc7484a94fc"
        "@onelake.dfs.fabric.microsoft.com/"
        "975b4954-52d5-4adb-baab-c6cd6452c8e5/Tables/fi_db_helloflow_customer",

    "fi_db_tx_entries":
        "abfss://b8650460-cb28-40ed-b824-740f3cd956f3"
        "@onelake.dfs.fabric.microsoft.com/"
        "f51e4b7c-79a8-47c3-91b7-5cbd92bf65b0/Tables/Fi_db_tx_entries",

    "cc_5749_call_logs":
        "abfss://4ee0ce3e-b5a2-474a-8a0f-1fc7484a94fc"
        "@onelake.dfs.fabric.microsoft.com/"
        "975b4954-52d5-4adb-baab-c6cd6452c8e5/Tables/cc_5749_call_logs",

    "cc_5749_customers":
        "abfss://4ee0ce3e-b5a2-474a-8a0f-1fc7484a94fc"
        "@onelake.dfs.fabric.microsoft.com/"
        "975b4954-52d5-4adb-baab-c6cd6452c8e5/Tables/cc_5749_customers",

    # <-- Lakehouse output tables, matching your old pattern exactly:
    "merchant_master":
        "abfss://Churn@onelake.dfs.fabric.microsoft.com/"
        "ml_ops.Lakehouse/Tables/Merchant_Master",

    "churn_features":
        "abfss://Churn@onelake.dfs.fabric.microsoft.com/"
        "ml_ops.Lakehouse/Tables/Churn_Features",

    "churn_predictions":
        "abfss://Churn@onelake.dfs.fabric.microsoft.com/"
        "ml_ops.Lakehouse/Tables/Churn_Predictions",

    # local artifacts
    "training_columns":
        "/lakehouse/default/Files/training_columns.pkl",
    "model_artifact":
        "/lakehouse/default/Files/churn_pipeline_tuned.joblib"
}
