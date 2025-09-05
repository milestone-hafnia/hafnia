import json
from pathlib import Path
from typing import Dict, List, Optional

import click

from hafnia import http
from hafnia.log import user_logger
from hafnia.utils import archive_dir, get_recipe_path, pretty_print_list_as_table, timed


@timed("Fetching dataset info.")
def get_dataset_id(dataset_name: str, endpoint: str, api_key: str) -> str:
    headers = {"Authorization": api_key}
    full_url = f"{endpoint}?name__iexact={dataset_name}"
    dataset_responses: List[Dict] = http.fetch(full_url, headers=headers)  # type: ignore[assignment]
    if not dataset_responses:
        raise ValueError(f"Dataset '{dataset_name}' was not found in the dataset library.")
    try:
        return dataset_responses[0]["id"]
    except (IndexError, KeyError) as e:
        raise ValueError("Dataset information is missing or invalid") from e


@timed("Get or create dataset recipe")
def get_or_create_dataset_recipe(
    recipe: dict,
    endpoint: str,
    api_key: str,
    name: Optional[str] = None,
) -> Optional[Dict]:
    headers = {"Authorization": api_key}
    data = {"template": {"body": recipe}}
    if name is not None:
        data["name"] = name  # type: ignore[assignment]
    response = http.post(endpoint, headers=headers, data=data)
    return response


def get_or_create_dataset_recipe_by_dataset_name(dataset_name: str, endpoint: str, api_key: str) -> Dict:
    return get_or_create_dataset_recipe(recipe=dataset_name, endpoint=endpoint, api_key=api_key)


def get_dataset_recipes(endpoint: str, api_key: str) -> List[Dict]:
    headers = {"Authorization": api_key}
    dataset_recipes: List[Dict] = http.fetch(endpoint, headers=headers)  # type: ignore[assignment]
    return dataset_recipes


def get_dataset_recipe_by_id(dataset_recipe_id: str, endpoint: str, api_key: str) -> Dict:
    headers = {"Authorization": api_key}
    full_url = f"{endpoint}/{dataset_recipe_id}"
    dataset_recipe_info: Dict = http.fetch(full_url, headers=headers)  # type: ignore[assignment]
    if not dataset_recipe_info:
        raise ValueError(f"Dataset recipe with ID '{dataset_recipe_id}' was not found.")
    return dataset_recipe_info


def get_or_create_dataset_recipe_from_path(
    path_recipe_json: Path, endpoint: str, api_key: str, name: Optional[str] = None
) -> Dict:
    path_recipe_json = Path(path_recipe_json)
    if not path_recipe_json.exists():
        raise click.ClickException(f"Dataset recipe file '{path_recipe_json}' does not exist.")
    json_dict = json.loads(path_recipe_json.read_text())
    return get_or_create_dataset_recipe(json_dict, endpoint=endpoint, api_key=api_key, name=name)


def delete_dataset_recipe_by_id(id: str, endpoint: str, api_key: str) -> Dict:
    headers = {"Authorization": api_key}
    full_url = f"{endpoint}/{id}"
    response = http.delete(endpoint=full_url, headers=headers)
    return response


def delete_dataset_recipe_by_name(name: str, endpoint: str, api_key: str) -> Optional[Dict]:
    recipe_response = get_dataset_recipe_by_name(name, endpoint=endpoint, api_key=api_key)

    if recipe_response:
        return delete_dataset_recipe_by_id(recipe_response["id"], endpoint=endpoint, api_key=api_key)
    return recipe_response


@timed("Get dataset recipe")
def get_dataset_recipe_by_name(name: str, endpoint: str, api_key: str) -> Optional[Dict]:
    headers = {"Authorization": api_key}
    full_url = f"{endpoint}?name__iexact={name}"
    dataset_recipes: List[Dict] = http.fetch(full_url, headers=headers)  # type: ignore[assignment]
    if len(dataset_recipes) == 0:
        return None

    if len(dataset_recipes) > 1:
        user_logger.warning(
            f"Found {len(dataset_recipes)} dataset recipes called '{name}' in your organization. Using the first one."
        )

    dataset_recipe = dataset_recipes[0]
    return dataset_recipe


@timed("Uploading recipe.")
def create_training_recipe(source_dir: Path, endpoint: str, api_key: str) -> str:
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
    response = http.post(endpoint, headers=headers, data=data, multipart=True)
    return response["id"]


@timed("Get training recipe.")
def get_training_recipe_by_id(id: str, endpoint: str, api_key: str) -> Dict:
    full_url = f"{endpoint}/{id}"
    headers = {"Authorization": api_key}
    response: Dict = http.fetch(full_url, headers=headers)  # type: ignore[assignment]
    return response


@timed("Get training recipes")
def get_training_recipes(endpoint: str, api_key: str) -> List[Dict]:
    headers = {"Authorization": api_key}
    recipes: List[Dict] = http.fetch(endpoint, headers=headers)  # type: ignore[assignment]
    return recipes


@timed("Fetching environment info.")
def get_environments(endpoint: str, api_key: str) -> List[Dict]:
    headers = {"Authorization": api_key}
    envs: List[Dict] = http.fetch(endpoint, headers=headers)  # type: ignore[assignment]
    return envs


def pretty_print_training_environments(envs: List[Dict]) -> None:
    ENV_FIELDS = {
        "Name": "name",
        "Instance": "instance",
        "GPU": "gpu",
        "GPU Count": "gpu_count",
        "GPU RAM": "vram",
        "CPU": "cpu",
        "CPU Count": "cpu_count",
        "RAM": "ram",
    }
    pretty_print_list_as_table(
        table_title="Available Training Environments",
        dict_items=envs,
        column_name_to_key_mapping=ENV_FIELDS,
    )


def get_exp_environment_id(name: str, endpoint: str, api_key: str) -> str:
    envs = get_environments(endpoint=endpoint, api_key=api_key)

    for env in envs:
        if env["name"] == name:
            return env["id"]

    pretty_print_training_environments(envs)

    available_envs = [env["name"] for env in envs]

    raise ValueError(f"Environment '{name}' not found. Available environments are: {available_envs}")


@timed("Creating experiment.")
def create_experiment(
    experiment_name: str,
    dataset_id: str,
    dataset_recipe_id: str,
    training_recipe_id: str,
    exec_cmd: str,
    environment_id: str,
    endpoint: str,
    api_key: str,
) -> Dict:
    headers = {"Authorization": api_key}
    response = http.post(
        endpoint,
        headers=headers,
        data={
            "name": experiment_name,
            "recipe": training_recipe_id,
            "dataset": dataset_id,
            "dataset_recipe": dataset_recipe_id,
            "command": exec_cmd,
            "environment": environment_id,
        },
    )
    return response
