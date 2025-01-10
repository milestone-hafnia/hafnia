import json
import os
from pathlib import Path
from shutil import rmtree
from typing import Dict


class Config:
    API_KEY: str = "MDI_API_KEY"
    ORGANIZATION_KEY: str = "organization_id"
    PLATFORM_URL_KEY: str = "platform_url"

    config_dir: str = ".mdi"
    config_file: str = "config.json"

    @property
    def api_key(self) -> str:
        if self.API_KEY in self.config:
            return self.config[self.API_KEY]
        if os.getenv("MDI_API_KEY_SECRET_NAME", None) is not None:
            return self.get_secret_value(os.getenv("MDI_API_KEY_SECRET_NAME"))
        print("API key not set. Please configure the cli with `mdi configure`.")
        exit(1)

    @property
    def organization_id(self) -> str:
        return self.config[self.ORGANIZATION_KEY] if self.ORGANIZATION_KEY in self.config else ""

    @organization_id.setter
    def organization_id(self, value: str) -> None:
        self.config[self.ORGANIZATION_KEY] = value
        self.update()

    @property
    def platform_url(self) -> str:
        return self.config[self.PLATFORM_URL_KEY] if self.PLATFORM_URL_KEY in self.config else ""

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

        self.config[self.API_KEY] = api_key
        self.update()

    def set_platform_url(self, url: str) -> None:
        base_url = url.rstrip("/")
        self.config[self.PLATFORM_URL_KEY] = base_url
        self.config["api"] = self.get_api_mapping(base_url)
        self.update()

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
        apis = self.config["api"]
        if method not in apis:
            raise ValueError(f"Method {method} not found in config.")
        return apis[method]

    def update(self) -> None:
        with open(self.config_file, "w") as f:
            json.dump(self.config, f, indent=4)

    def clear(self) -> None:
        self.config = {}
        rmtree(self.config_dir)

    def get_secret_value(self, secret_name: str) -> str:
        import boto3

        with boto3.Session() as session:
            client = session.client("secretsmanager", aws_region=os.getenv("AWS_REGION"))
            response = client.get_secret_value(SecretId=secret_name)
            return response["SecretString"]


CONFIG = Config()
