import os
from configparser import ConfigParser, SectionProxy
from pathlib import Path
from typing import Optional

import keyring
from xdg_base_dirs import xdg_cache_home, xdg_config_home

# Configurable ENV variables.
# Note: the readme should be updated if adding/editing these variables.
MDI_API_KEY = "MDI_API_KEY"
MDI_API_URL = "MDI_API_URL"
MDI_CONFIG_FILE = "MDI_CONFIG_FILE"
MDI_CACHE_DIR = "MDI_CACHE_DIR"
# End of configurable ENV variables.

MDI_FOLDER_NAME = "mdi"
MDI_CONFIG_FILE = Path(
    os.environ.get(MDI_CONFIG_FILE, default=xdg_config_home() / MDI_FOLDER_NAME / "config.ini")
)
MDI_CACHE_DIR = Path(os.environ.get(MDI_CACHE_DIR, default=xdg_cache_home() / MDI_FOLDER_NAME))
DEFAULT_API_URL = "https://api.mdi.milestonesys.com"

KEY_STORE_KEYRING = "keyring"
KEY_STORE_CONFIG = "config-file"
ALLOWED_KEY_STORES = [KEY_STORE_KEYRING, KEY_STORE_CONFIG]


class Config:
    _KEY_CURRENT_PROFILE = "current_profile"
    _SECTION_PROFILE_PREFIX = "profile."

    def __init__(self, config_file: Path, cache_dir: Path) -> None:
        self._config_file = config_file
        self._config = self._read_config(config_file, cache_dir)

    def _read_config(self, config_file: Path, cache_dir: Path) -> ConfigParser:
        config = ConfigParser()
        config.read_dict(self._default_config(cache_dir))
        config.read(config_file)
        return config

    def _default_config(self, cache_dir: Path) -> dict:
        return {
            "DEFAULT": {
                "cache_dir": cache_dir,
                self._KEY_CURRENT_PROFILE: "",
            },
        }

    def get_api_url(self) -> Optional[str]:
        """Get API URL from the ENV variable or current profile."""
        if value := os.environ.get(MDI_API_URL):
            return value
        api_url = self._get_current_profile_value("api_url")
        if api_url:
            return api_url.rstrip("/")
        return None

    def get_api_key(self) -> Optional[str]:
        """Get API key from the ENV variable or current profile."""
        if value := os.environ.get(MDI_API_KEY):
            return value
        current_profile = self.get_current_profile()
        if not current_profile:
            return None
        key_store = current_profile.get("key_store", KEY_STORE_KEYRING)
        if key_store == KEY_STORE_KEYRING:
            return keyring.get_password("mdi", current_profile.name)
        else:
            return current_profile.get("api_key")

    def get_module_dir(self) -> Path:
        cache_dir = Path(self._get_default_config_value("cache_dir"))
        return cache_dir / "custom" / "datasets_modules"

    def get_current_profile(self) -> Optional[SectionProxy]:
        current_profile_name = self.get_current_profile_name()
        if not current_profile_name:
            return None
        return self.get_profile(current_profile_name)

    def get_current_profile_name(self) -> Optional[str]:
        default_section = self._config["DEFAULT"]
        if current_profile := default_section.get(self._KEY_CURRENT_PROFILE):
            return current_profile.removeprefix(self._SECTION_PROFILE_PREFIX)
        return None

    def _get_default_config_value(self, key: str) -> str:
        return self._config["DEFAULT"].get(key)

    def _get_current_profile_value(self, key: str) -> Optional[str]:
        default_section = self._config["DEFAULT"]
        if current_profile := default_section.get(self._KEY_CURRENT_PROFILE):
            if current_profile and current_profile in self._config:
                return self._config[current_profile].get(key, "")
        return None

    def list_profiles(self) -> dict[str, str]:
        name_urls = {}
        for section in self._config:
            if section.startswith(self._SECTION_PROFILE_PREFIX):
                name_urls[section.removeprefix(self._SECTION_PROFILE_PREFIX)] = self._config[
                    section
                ].get("api_url", "")

        return name_urls

    def get_profile(self, name: str) -> Optional[SectionProxy]:
        profile_section = self._profile_name_to_section(name)
        if profile_section in self._config:
            return self._config[profile_section]
        return None

    def update_profile_api_key(self, profile_name: str, value: str) -> None:
        profile = self.get_profile(profile_name)
        if not profile:
            raise ValueError(f"profile with name '{profile_name}' does not exists")
        profile_section_name = self._profile_name_to_section(profile_name)
        if profile.get("key_store") == KEY_STORE_KEYRING:
            keyring.set_password("mdi", profile_section_name, value)
        else:
            self._config.set(profile_section_name, "api_key", value)
            self.save_config()

    def update_profile_api_url(self, profile_name: str, value: str) -> None:
        self._update_profile_value(profile_name, "api_url", value)

    def _update_profile_value(self, profile_name: str, key: str, value: str) -> None:
        if not self.get_profile(profile_name):
            raise ValueError(f"profile with name '{profile_name}' does not exists")
        profile_section = self._profile_name_to_section(profile_name)
        self._config.set(profile_section, key, value)
        self.save_config()

    def create_profile(
        self, name: str, api_url: str, api_key: str, key_store: str = KEY_STORE_KEYRING
    ) -> None:
        if self.get_profile(name):
            raise ValueError(f"profile with name '{name}' already exists")
        if key_store not in ALLOWED_KEY_STORES:
            raise ValueError(
                f"key store '{key_store}' is not supported. Supported options:"
                f" {ALLOWED_KEY_STORES}"
            )

        profile = {"api_url": api_url, "key_store": KEY_STORE_KEYRING}
        if key_store == KEY_STORE_CONFIG:
            profile["key_store"] = KEY_STORE_CONFIG
            profile["api_key"] = api_key
        profile_section = self._profile_name_to_section(name)
        self._config[profile_section] = profile
        self.save_config()
        if key_store == KEY_STORE_KEYRING:
            try:
                keyring.set_password("mdi", profile_section, api_key)
            except keyring.errors.KeyringError:
                # Remove the created profile if we fail to store the api_key.
                del self._config[profile_section]
                self.save_config()
                raise

    def delete_profile(self, name: str) -> None:
        if not self.get_profile(name):
            raise ValueError(f"profile '{name}' does not exists")

        profile_section = self._profile_name_to_section(name)
        key_store = self._config[profile_section].get("key_store")
        del self._config[profile_section]
        # If this was the current profile, remove the reference.
        default_section = self._config["DEFAULT"]
        if default_section[self._KEY_CURRENT_PROFILE] == profile_section:
            default_section[self._KEY_CURRENT_PROFILE] = ""

        self.save_config()
        if key_store == KEY_STORE_KEYRING:
            keyring.delete_password("mdi", profile_section)

    def set_current_profile(self, name: str) -> None:
        if not self.get_profile(name):
            raise ValueError(f"profile '{name}' does not exist")

        profile_section = self._profile_name_to_section(name)
        self._config["DEFAULT"][self._KEY_CURRENT_PROFILE] = profile_section
        self.save_config()

    def save_config(self) -> None:
        config_folder = self._config_file.parent
        # Make sure the folder exists
        if not config_folder.exists():
            config_folder.mkdir(parents=True, exist_ok=True)

        with open(self._config_file, "w") as f:
            self._config.write(f)

        # Set the file permissions to 600
        os.chmod(self._config_file, 0o600)

    def _profile_name_to_section(self, name: str) -> str:
        return f"{self._SECTION_PROFILE_PREFIX}{name}"


config = Config(MDI_CONFIG_FILE, MDI_CACHE_DIR)
