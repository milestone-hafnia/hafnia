from pathlib import Path
from typing import Dict, List, Optional

import click

from cli.config import Config
from hafnia import http
from hafnia.platform.dataset_recipe import (
    get_dataset_recipe_by_id,
    get_dataset_recipe_by_name,
    get_or_create_dataset_recipe_by_dataset_name,
)
from hafnia.platform.train_recipe import create_training_recipe
from hafnia.utils import pretty_print_list_as_table, timed


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


def get_dataset_recipe_by_dataset_identifies(
    cfg: Config,
    dataset_name: Optional[str],
    dataset_recipe_name: Optional[str],
    dataset_recipe_id: Optional[str],
) -> Dict:
    dataset_identifiers = [dataset_name, dataset_recipe_name, dataset_recipe_id]
    n_dataset_identifies_defined = sum([bool(identifier) for identifier in dataset_identifiers])

    if n_dataset_identifies_defined > 1:
        raise click.ClickException(
            "Multiple dataset identifiers have been provided. Define only one dataset identifier."
        )

    dataset_recipe_endpoint = cfg.get_platform_endpoint("dataset_recipes")
    if dataset_name:
        return get_or_create_dataset_recipe_by_dataset_name(dataset_name, dataset_recipe_endpoint, cfg.api_key)

    if dataset_recipe_name:
        recipe = get_dataset_recipe_by_name(dataset_recipe_name, dataset_recipe_endpoint, cfg.api_key)
        if recipe is None:
            raise click.ClickException(f"Dataset recipe '{dataset_recipe_name}' was not found in the dataset library.")
        return recipe

    if dataset_recipe_id:
        return get_dataset_recipe_by_id(dataset_recipe_id, dataset_recipe_endpoint, cfg.api_key)

    raise click.MissingParameter(
        "At least one dataset identifier must be provided. Set one of the following:\n"
        "  --dataset <name>  -- E.g. '--dataset mnist'\n"
        "  --dataset-recipe <name>  -- E.g. '--dataset-recipe my-recipe'\n"
        "  --dataset-recipe-id <id>  -- E.g. '--dataset-recipe-id 5e454c0d-fdf1-4d1f-9732-771d7fecd28e'\n"
    )


def get_training_recipe_by_identifies(
    cfg: Config,
    train_recipe_path: Optional[Path],
    train_recipe_id: Optional[str],
) -> str:
    from hafnia.platform import get_training_recipe_by_id

    if train_recipe_path is not None and train_recipe_id is not None:
        raise click.ClickException("Multiple training recipe identifiers have been provided. Define only one.")

    if train_recipe_path is not None:
        train_recipe_path = Path(train_recipe_path)
        if not train_recipe_path.exists():
            raise click.ClickException(f"Training recipe path '{train_recipe_path}' does not exist.")
        recipe_id = create_training_recipe(
            train_recipe_path,
            cfg.get_platform_endpoint("training_recipes"),
            cfg.api_key,
        )
        return recipe_id

    if train_recipe_id:
        train_recipe = get_training_recipe_by_id(
            id=train_recipe_id, endpoint=cfg.get_platform_endpoint("training_recipes"), api_key=cfg.api_key
        )
        return train_recipe["id"]

    raise click.MissingParameter(
        "At least one training recipe identifier must be provided. Set one of the following:\n"
        "  --train-recipe-path <path>  -- E.g. '--train-recipe-path .'\n"
        "  --train-recipe-id <id>  -- E.g. '--train-recipe-id 5e454c0d-fdf1-4d1f-9732-771d7fecd28e'\n"
    )
