import notebookutils

def get_storage_options():
    """
    Returns the storage options dict for token-based Lakehouse access in delta-rs.
    """
    token = notebookutils.credentials.getToken("storage")

    return {
        # -------------------------------
        # REQUIRED for delta-rs on ADLS:
        "account_name":        "onelake",        # your Fabric account
        # (alias supported: "azure_storage_account_name")

        # the bearer token from the notebook
        "bearer_token":        token,

        # tells it to build URLs against *.dfs.fabric.microsoft.com
        "use_fabric_endpoint": "true"
    }
