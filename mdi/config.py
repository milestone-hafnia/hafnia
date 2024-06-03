import os
from configparser import ConfigParser, SectionProxy
from pathlib import Path
from typing import Optional

from xdg_base_dirs import xdg_cache_home, xdg_config_home, xdg_data_home

DEFAULT_API_URL = "https://api.mdi.milestonesys.com"
MDI_FOLDER_NAME = "mdi"
MDI_CONFIG_FILE = Path(os.environ.get("MDI_CONFIG_FILE", default=xdg_config_home() / MDI_FOLDER_NAME / "config.ini"))
MDI_HOME_DIR = xdg_data_home() / MDI_FOLDER_NAME
MDI_CACHE_DIR = xdg_cache_home() / MDI_FOLDER_NAME


class Config:
    _KEY_CURRENT_PROFILE = "current_profile"
    _SECTION_PROFILE_PREFIX = "profile."

    def __init__(self, config_file: Path, home_dir: Path, cache_dir: Path) -> None:
        self._config_file = config_file
        self._config = self._read_config(config_file, home_dir, cache_dir)

    def _read_config(self, config_path: Path, home_dir: Path, cache_dir: Path) -> ConfigParser:
        config = ConfigParser()
        config.read_dict(self._default_config(home_dir, cache_dir))
        config.read(config_path)
        return config

    def _default_config(self, home_dir: Path, cache_dir: Path) -> dict:
        return {
            "DEFAULT": {
                "home": home_dir,
                "cache_dir": cache_dir,
                self._KEY_CURRENT_PROFILE: "",
            },
        }

    def get_api_url(self) -> Optional[str]:
        """Get API URL from the ENV variable or config file."""
        api_url = self._get_current_profile_value("api_url")
        if api_url:
            return api_url.rstrip("/")
        return None

    def get_api_key(self) -> Optional[str]:
        """Get API key from the ENV variable or config file."""
        return self._get_current_profile_value("api_key")

    def get_module_dir(self) -> Path:
        cache_dir = Path(self._get_config_value("cache_dir"))
        return cache_dir / "custom" / "datasets_modules"

    def get_current_profile_name(self) -> Optional[str]:
        default_section = self._config["DEFAULT"]
        if current_profile := default_section.get(self._KEY_CURRENT_PROFILE):
            return current_profile.removeprefix(self._SECTION_PROFILE_PREFIX)

        return None

    def _get_config_value(self, key: str) -> str:
        return os.environ.get(f"MDI_{key.upper()}", default=self._config["DEFAULT"].get(key))

    def _get_current_profile_value(self, key: str) -> Optional[str]:
        if value := os.environ.get(f"MDI_{key.upper()}"):
            return value

        default_section = self._config["DEFAULT"]
        if current_profile := default_section.get(self._KEY_CURRENT_PROFILE):
            if current_profile and current_profile in self._config:
                return self._config[current_profile].get(key, "")

        return None

    def list_profiles(self) -> dict[str, str]:
        name_urls = {}
        for section in self._config:
            if section.startswith(self._SECTION_PROFILE_PREFIX):
                name_urls[section.removeprefix(self._SECTION_PROFILE_PREFIX)] = self._config[section].get("api_url", "")

        return name_urls

    def get_profile(self, name: str) -> Optional[SectionProxy]:
        profile_section = self._profile_name_to_section(name)
        if profile_section in self._config:
            return self._config[profile_section]
        return None

    def update_profile_values(self, profile_name: str, key_vals: dict[str, str]) -> None:
        if not self.get_profile(profile_name):
            raise ValueError(f"profile with name '{profile_name}' does not exists")

        profile_section = self._profile_name_to_section(profile_name)
        for key, value in key_vals.items():
            self._config.set(profile_section, key, value)

        self.save_config()

    def create_profile(self, name: str, api_url: str, api_key: str) -> None:
        if self.get_profile(name):
            raise ValueError(f"profile with name '{name}' already exists")

        profile_section = self._profile_name_to_section(name)
        self._config[profile_section] = {"api_url": api_url, "api_key": api_key}
        self.save_config()

    def delete_profile(self, name: str) -> None:
        if not self.get_profile(name):
            raise ValueError(f"profile '{name}' does not exists")

        profile_section = self._profile_name_to_section(name)
        del self._config[profile_section]

        # If this was the current profile, remove the reference.
        default_section = self._config["DEFAULT"]
        if default_section[self._KEY_CURRENT_PROFILE] == profile_section:
            default_section[self._KEY_CURRENT_PROFILE] = ""

        self.save_config()

    def set_current_profile(self, name: str) -> None:
        if not self.get_profile(name):
            raise ValueError(f"profile '{name}' does not exist")

        profile_section = self._profile_name_to_section(name)
        self._config["DEFAULT"][self._KEY_CURRENT_PROFILE] = profile_section
        self.save_config()

    def save_config(self) -> None:
        config_folder = self._config_file.parent
        # Make sure the .mdi folder exists
        if not config_folder.exists():
            config_folder.mkdir(parents=True, exist_ok=True)

        with open(self._config_file, "w") as f:
            self._config.write(f)

        # Set the file permissions to 600
        os.chmod(self._config_file, 0o600)

    def _profile_name_to_section(self, name: str) -> str:
        return f"{self._SECTION_PROFILE_PREFIX}{name}"


config = Config(MDI_CONFIG_FILE, MDI_HOME_DIR, MDI_CACHE_DIR)
