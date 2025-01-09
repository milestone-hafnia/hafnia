import json
from pathlib import Path
from typing import Optional

from mdi_python_tools.log import logger
from mdi_python_tools.mdi_sdk import fetch, get_api_mapping


class Config:
    config_dir: str = ".mdi"
    config_file: str = "config.json"

    def __init__(self):
        self.config_dir = Path.home() / self.config_dir
        self.config_dir.mkdir(exist_ok=True)
        self.config_file = self.config_dir / self.config_file
        self.config = self._load_config() if self.config_file.exists() else {}
        self.get_organization_id(self.config["api"]["organizations"])

    def _load_config(self) -> dict:
        """Load configuration from file."""
        try:
            with open(self.config_file, "r") as f:
                return json.load(f)
        except json.JSONDecodeError:
            return {}

    def set_api_key(self, api_key: str) -> None:
        """
        Set the MDI API key.

        Args:
            api_key (str): The API key to store
        """
        if not api_key:
            raise ValueError("API key cannot be empty")

        self.config["MDI_API_KEY"] = api_key
        self.update()

    def set_platform_url(self, url: str) -> None:
        base_url = url.rstrip("/")
        self.config["api"] = get_api_mapping(base_url)
        self.config["organization_id"] = self.get_organization_id(
            self.config["api"]["organizations"]
        )
        self.update()

    def get_organization_id(self, endpoint: str) -> Optional[str]:
        headers = {"X-APIKEY": self.get_api_key()}
        try:
            org_info = fetch(endpoint, headers=headers)
        except Exception as e:
            logger.error(f"Error fetching organization info: {str(e)}")
            return None
        return org_info[0]["id"]

    def get_api_key(self) -> Optional[str]:
        """
        Get the stored MDI API key.

        Returns:
            Optional[str]: The stored API key or None if not set
        """
        return self.config.get("MDI_API_KEY")

    def get_platform_endpoint(self, func: str, method: str) -> str:
        if func not in self.config:
            raise ValueError(f"Function {func} not found in config.")
        if method not in self.config[func]:
            raise ValueError(f"Method {method} not found in config.")
        return self.config[func][method]

    def delete_api_key(self) -> None:
        """Remove the stored API key."""
        config = self._load_config()
        if "MDI_API_KEY" in config:
            del config["MDI_API_KEY"]
            self.update()

    def update(self) -> None:
        with open(self.config_file, "w") as f:
            json.dump(self.config, f, indent=4)


CONFIG = Config()
