import os
from configparser import ConfigParser
from pathlib import Path

from xdg_base_dirs import xdg_cache_home, xdg_config_home, xdg_data_home

MDI_FOLDER_NAME = "mdi"
DEFAULT_CONFIG = {
    "mdi": {
        "api_url": "https://api.mdi.milestonesys.com",
        "home": xdg_data_home() / MDI_FOLDER_NAME,
        "cache_dir": xdg_cache_home() / MDI_FOLDER_NAME,
    }
}


def read_config(config_file_path: Path) -> dict:
    config = ConfigParser()
    config.read_dict(DEFAULT_CONFIG)
    config.read(config_file_path)
    return config["mdi"]


def _get_config_value(key: str, config: dict) -> str:
    """Get config value either from environment or config file."""
    return os.environ.get(f"MDI_{key.upper()}", default=config.get(key))


MDI_CONFIG_FILE = Path(
    os.environ.get(
        "MDI_CONFIG_FILE", default=xdg_config_home() / MDI_FOLDER_NAME / "config.ini"
    )
)
_config = read_config(MDI_CONFIG_FILE)

MDI_API_URL = _get_config_value("api_url", _config)
MDI_API_TEMP_CREDS = (
    f"{MDI_API_URL}/api/v1/datasets/{{obj_id}}/get_temporary_credentials/"
)
MDI_API_DATASETS = f"{MDI_API_URL}/api/v1/datasets/?name__iexact={{name}}"
MDI_API_DATASETS_BY_ID = f"{MDI_API_URL}/api/v1/datasets/{{id}}/"

MDI_HOME = Path(_get_config_value("home", _config))
MDI_CREDENTIALS = MDI_HOME / "credentials"

MDI_CACHE_DIR = Path(_get_config_value("cache_dir", _config))
MDI_MODULE_DIR = MDI_CACHE_DIR / "custom" / "datasets_modules"
