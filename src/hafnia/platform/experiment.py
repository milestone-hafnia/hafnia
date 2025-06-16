from pathlib import Path
from typing import Optional

from hafnia.http import fetch, post
from hafnia.log import user_logger
from hafnia.utils import archive_dir, get_recipe_path, timed


@timed("Fetching dataset info.")
def get_dataset_id(dataset_name: str, endpoint: str, api_key: str) -> str:
    headers = {"Authorization": api_key}
    full_url = f"{endpoint}?name__iexact={dataset_name}"
    dataset_info = fetch(full_url, headers=headers)
    if not dataset_info:
        raise ValueError(f"Dataset '{dataset_name}' was not found in the dataset library.")
    try:
        return dataset_info[0]["id"]
    except (IndexError, KeyError) as e:
        raise ValueError("Dataset information is missing or invalid") from e


@timed("Uploading recipe.")
def create_recipe(source_dir: Path, endpoint: str, api_key: str) -> str:
    source_dir = source_dir.resolve()  # Ensure the path is absolute to handle '.' paths are given an appropriate name.
    path_recipe = get_recipe_path(recipe_name=source_dir.name)
    zip_path = archive_dir(source_dir, output_path=path_recipe)
    user_logger.info(f"Recipe created and stored in '{path_recipe}'")

    headers = {"Authorization": api_key, "accept": "application/json"}
    data = {
        "name": path_recipe.name,
        "description": "Recipe created by Hafnia CLI",
        "file": (zip_path.name, Path(zip_path).read_bytes()),
    }
    response = post(endpoint, headers=headers, data=data, multipart=True)
    return response["id"]


@timed("Fetching environment info.")
def get_exp_environment_id(name: str, endpoint: str, api_key: str) -> str:
    headers = {"Authorization": api_key}
    env_info = fetch(endpoint, headers=headers)

    for env in env_info:
        if env["name"] == name:
            return env["id"]

    raise ValueError(f"Environment '{name}' not found")


@timed("Creating exepriment.")
def create_experiment(
    exp_name: str,
    dataset_id: str,
    recipe_id: str,
    exec_cmd: str,
    environment_id: str,
    endpoint: str,
    api_key: str,
) -> Optional[str]:
    headers = {"Authorization": api_key}
    response = post(
        endpoint,
        headers=headers,
        data={
            "name": exp_name,
            "recipe": recipe_id,
            "dataset": dataset_id,
            "command": exec_cmd,
            "environment": environment_id,
        },
    )
    return response["id"]
