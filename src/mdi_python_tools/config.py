import json
from pathlib import Path
from typing import Optional


class Config:
    config_dir: str = ".mdi"
    config_file: str = "config.json"

    def __init__(self):
        self.config_dir = Path.home() / self.config_dir
        self.config_dir.mkdir(exist_ok=True)
        self.config_file = self.config_dir / self.config_file
        self.config = self._load_config() if self.config_file.exists() else {}

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
        self.config["experiment"] = {
            "create": f"{url}/api/v1/experiments",
            "list": f"{url}/api/v1/experiments",
            "status": f"{url}/api/v1/experiments",
        }
        self.update()

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
