import json
import os
from pathlib import Path
from typing import Dict, List, Optional

from pydantic import BaseModel, field_validator

import cli.consts as consts
from hafnia.log import logger


class ConfigSchema(BaseModel):
    organization_id: str = ""
    platform_url: str = ""
    api_key: Optional[str] = None
    api_mapping: Optional[Dict[str, str]] = None

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
    def organization_id(self) -> str:
        return self.config.organization_id

    @organization_id.setter
    def organization_id(self, value: str) -> None:
        self.config.organization_id = value

    @property
    def platform_url(self) -> str:
        return self.config.platform_url

    @platform_url.setter
    def platform_url(self, value: str) -> None:
        base_url = value.rstrip("/")
        self.config.platform_url = base_url
        self.config.api_mapping = self.get_api_mapping(base_url)

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
        self.config_data.profiles[profile_name] = profile
        if set_active:
            self.config_data.active_profile = profile_name
        self.save_config()

    def get_api_mapping(self, base_url: str) -> Dict:
        return {
            "organizations": f"{base_url}/api/v1/organizations",
            "recipes": f"{base_url}/api/v1/recipes",
            "experiments": f"{base_url}/api/v1/experiments",
            "experiment_environments": f"{base_url}/api/v1/experiment-environments",
            "experiment_runs": f"{base_url}/api/v1/experiment-runs",
            "runs": f"{base_url}/api/v1/experiments-runs",
            "datasets": f"{base_url}/api/v1/datasets",
        }

    def get_platform_endpoint(self, method: str) -> str:
        """Get specific API endpoint"""
        if not self.config.api_mapping or method not in self.config.api_mapping:
            raise ValueError(f"{method} is not supported.")
        return self.config.api_mapping[method]

    def load_config(self) -> ConfigFileSchema:
        """Load configuration from file."""
        if not self.config_path.exists():
            return ConfigFileSchema()
        try:
            with open(self.config_path.as_posix(), "r") as f:
                data = json.load(f)
            return ConfigFileSchema(**data)
        except json.JSONDecodeError:
            logger.error("Error decoding JSON file.")
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
