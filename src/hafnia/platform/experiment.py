from pathlib import Path
from typing import Optional

from hafnia.http import fetch, post
from hafnia.platform.builder import validate_recipe
from hafnia.utils import archive_dir, get_recipe_path


def get_dataset_id(dataset_name: str, endpoint: str, api_key: str) -> Optional[str]:
    headers = {"X-APIKEY": api_key}
    full_url = f"{endpoint}?name__iexact={dataset_name}"
    dataset_info = fetch(full_url, headers=headers)
    if not dataset_info:
        raise ValueError(f"Dataset '{dataset_name}' was not found in the dataset library.")
    return dataset_info[0]["id"]


def create_recipe(source_dir: Path, endpoint: str, api_key: str, organization_id: str) -> Optional[str]:
    headers = {"X-APIKEY": api_key, "accept": "application/json"}
    path_recipe = get_recipe_path(recipe_name=source_dir.name)
    zip_path = archive_dir(source_dir, output_path=path_recipe)

    print(f"Recipe created and stored in '{path_recipe}'")

    validate_recipe(zip_path)
    with open(zip_path, "rb") as zip_file:
        fields = {
            "name": path_recipe.name,
            "description": "Recipe created by Hafnia CLI",
            "organization": organization_id,
            "file": (zip_path.name, zip_file.read()),
        }
        response = post(endpoint, headers=headers, data=fields, multipart=True)
        return response["id"]


def get_exp_environment_id(name: str, endpoint: str, api_key: str) -> Optional[str]:
    headers = {"X-APIKEY": api_key}
    env_info = fetch(endpoint, headers=headers)
    return next((env["id"] for env in env_info if env["name"] == name), None)


def create_experiment(
    exp_name: str,
    dataset_id: str,
    recipe_id: str,
    exec_cmd: str,
    environment_id: str,
    endpoint: str,
    api_key: str,
    organization_id: str,
) -> Optional[str]:
    headers = {"X-APIKEY": api_key}
    response = post(
        endpoint,
        headers=headers,
        data={
            "organization": organization_id,
            "name": exp_name,
            "recipe": recipe_id,
            "dataset": dataset_id,
            "command": exec_cmd,
            "environment": environment_id,
        },
    )
    return response["id"]
