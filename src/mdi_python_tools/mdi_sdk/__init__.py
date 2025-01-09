from typing import Dict

from mdi_python_tools.mdi_sdk.client import fetch, post
from mdi_python_tools.mdi_sdk.resources import download_resource

__all__ = ["fetch", "post", "download_resource"]


def get_api_mapping(base_url: str) -> Dict:
    return {
        "organizations": f"{base_url}/api/v1/organizations",
        "recipes": f"{base_url}/api/v1/recipes",
        "experiments": f"{base_url}/api/v1/experiments",
        "experiment-environments": f"{base_url}/api/v1/experiment-environments",
        "runs": f"{base_url}/api/v1/experiments-runs",
        "datasets": f"{base_url}/api/v1/datasets",
    }
