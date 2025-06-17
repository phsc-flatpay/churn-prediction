import notebookutils

def get_storage_options():
    """
    Returns the storage options dict for token-based Lakehouse access in delta-rs
    """
    token = notebookutils.credentials.getToken("storage")

    return {
        # REQUIRED for delta-rs on ADLS:
        "account_name":        "onelake",        # Fabric account

        # the bearer token from the notebook
        "bearer_token":        token,
      
        "use_fabric_endpoint": "true"
    }
