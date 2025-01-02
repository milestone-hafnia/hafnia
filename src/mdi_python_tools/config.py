import json
from pathlib import Path
from typing import Optional


class Config:
    config_dir: str = ".mdi"
    config_file: str = "config.json"

    def __init__(self):
        self.config_dir = Path.home() / self.config_dir
        self.config_file = self.config_dir / self.config_file
        self._ensure_config_dir()

    def _ensure_config_dir(self) -> None:
        """Create config directory if it doesn't exist."""
        self.config_dir.mkdir(exist_ok=True)
        if not self.config_file.exists():
            self._save_config({})

    def _load_config(self) -> dict:
        """Load configuration from file."""
        try:
            with open(self.config_file, "r") as f:
                return json.load(f)
        except json.JSONDecodeError:
            return {}

    def _save_config(self, config: dict) -> None:
        """Save configuration to file."""
        with open(self.config_file, "w") as f:
            json.dump(config, f, indent=4)

    def set_api_key(self, api_key: str) -> None:
        """
        Set the MDI API key.

        Args:
            api_key (str): The API key to store
        """
        if not api_key:
            raise ValueError("API key cannot be empty")

        config = self._load_config()
        config["MDI_API_KEY"] = api_key
        self._save_config(config)

    def get_api_key(self) -> Optional[str]:
        """
        Get the stored MDI API key.

        Returns:
            Optional[str]: The stored API key or None if not set
        """
        config = self._load_config()
        return config.get("MDI_API_KEY")

    def delete_api_key(self) -> None:
        """Remove the stored API key."""
        config = self._load_config()
        if "MDI_API_KEY" in config:
            del config["MDI_API_KEY"]
            self._save_config(config)


CONFIG = Config()
