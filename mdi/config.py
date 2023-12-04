from xdg_base_dirs import xdg_cache_home, xdg_data_home

MDI_API_URL = "https://api.staging02.mdi.milestonesys.com"
MDI_API_TEMP_CREDS = (
    f"{MDI_API_URL}/api/v1/datasets/{{obj_id}}/get_temporary_credentials/"
)
MDI_API_DATASETS = f"{MDI_API_URL}/api/v1/datasets/?name__iexact={{name}}"
MDI_API_DATASETS_BY_ID = f"{MDI_API_URL}/api/v1/datasets/{{id}}/"
MDI_FOLDER_NAME = "mdi"
MDI_HOME = xdg_data_home() / MDI_FOLDER_NAME
MDI_CREDENTIALS = MDI_HOME / "credentials"
MDI_CACHE_DIR = xdg_cache_home() / MDI_FOLDER_NAME
MDI_MODULE_DIR = MDI_CACHE_DIR / "custom" / "datasets_modules"
