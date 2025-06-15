import json
import os
from pathlib import Path
from typing import Dict, List, Optional

from pydantic import BaseModel, field_validator

import cli.consts as consts
from hafnia.log import user_logger

PLATFORM_API_MAPPING = {
    "recipes": "/api/v1/recipes",
    "experiments": "/api/v1/experiments",
    "experiment_environments": "/api/v1/experiment-environments",
    "experiment_runs": "/api/v1/experiment-runs",
    "runs": "/api/v1/experiments-runs",
    "datasets": "/api/v1/datasets",
}


class ConfigSchema(BaseModel):
    platform_url: str = ""
    api_key: Optional[str] = None

    @field_validator("api_key")
    def validate_api_key(cls, value: str) -> str:
        if value is not None and len(value) < 10:
            raise ValueError("API key is too short.")
        return value


class ConfigFileSchema(BaseModel):
    active_profile: Optional[str] = None
    profiles: Dict[str, ConfigSchema] = {}


class Config:
    @property
    def available_profiles(self) -> List[str]:
        return list(self.config_data.profiles.keys())

    @property
    def active_profile(self) -> str:
        if self.config_data.active_profile is None:
            raise ValueError(consts.ERROR_PROFILE_NOT_EXIST)
        return self.config_data.active_profile

    @active_profile.setter
    def active_profile(self, value: str) -> None:
        profile_name = value.strip()
        if profile_name not in self.config_data.profiles:
            raise ValueError(f"Profile '{profile_name}' does not exist.")
        self.config_data.active_profile = profile_name

    @property
    def config(self) -> ConfigSchema:
        if not self.config_data.active_profile:
            raise ValueError(consts.ERROR_PROFILE_NOT_EXIST)
        return self.config_data.profiles[self.config_data.active_profile]

    @property
    def api_key(self) -> str:
        if self.config.api_key is not None:
            return self.config.api_key
        raise ValueError(consts.ERROR_API_KEY_NOT_SET)

    @api_key.setter
    def api_key(self, value: str) -> None:
        self.config.api_key = value

    @property
    def platform_url(self) -> str:
        return self.config.platform_url

    @platform_url.setter
    def platform_url(self, value: str) -> None:
        base_url = value.rstrip("/")
        self.config.platform_url = base_url

    def __init__(self, config_path: Optional[Path] = None) -> None:
        self.config_path = self.resolve_config_path(config_path)
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        self.config_data = self.load_config()

    def resolve_config_path(self, path: Optional[Path] = None) -> Path:
        if path:
            return Path(path).expanduser()

        config_env_path = os.getenv("MDI_CONFIG_PATH")
        if config_env_path:
            return Path(config_env_path).expanduser()

        return Path.home() / ".hafnia" / "config.json"

    def add_profile(self, profile_name: str, profile: ConfigSchema, set_active: bool = False) -> None:
        profile_name = profile_name.strip()
        if profile_name in self.config_data.profiles:
            user_logger.warning(
                f"Profile with name '{profile_name}' already exists, it will be overwritten by the new one."
            )

        self.config_data.profiles[profile_name] = profile
        if set_active:
            self.config_data.active_profile = profile_name
        self.save_config()

    def get_platform_endpoint(self, method: str) -> str:
        """Get specific API endpoint"""
        if method not in PLATFORM_API_MAPPING:
            raise ValueError(f"'{method}' is not supported.")
        endpoint = self.config.platform_url + PLATFORM_API_MAPPING[method]
        return endpoint

    def load_config(self) -> ConfigFileSchema:
        """Load configuration from file."""
        if not self.config_path.exists():
            return ConfigFileSchema()
        try:
            with open(self.config_path.as_posix(), "r") as f:
                data = json.load(f)
            return ConfigFileSchema(**data)
        except json.JSONDecodeError:
            user_logger.error("Error decoding JSON file.")
            raise ValueError("Failed to parse configuration file")

    def save_config(self) -> None:
        with open(self.config_path, "w") as f:
            json.dump(self.config_data.model_dump(), f, indent=4)

    def remove_profile(self, profile_name: str) -> None:
        if profile_name not in self.config_data.profiles:
            raise ValueError(f"Profile '{profile_name}' does not exist.")
        del self.config_data.profiles[profile_name]
        self.save_config()

    def is_configured(self) -> bool:
        return self.config_data.active_profile is not None

    def clear(self) -> None:
        self.config_data = ConfigFileSchema(active_profile=None, profiles={})
        if self.config_path.exists():
            self.config_path.unlink()
