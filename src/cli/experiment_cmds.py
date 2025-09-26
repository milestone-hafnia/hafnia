from pathlib import Path
from typing import Dict, Optional

import click

from cli.config import Config
from hafnia import utils
from hafnia.platform.dataset_recipe import (
    get_dataset_recipe_by_id,
    get_dataset_recipe_by_name,
    get_or_create_dataset_recipe_by_dataset_name,
)
from hafnia.platform.trainer_package import create_trainer_package


@click.group(name="experiment")
def experiment() -> None:
    """Experiment management commands"""
    pass


@experiment.command(name="environments")
@click.pass_obj
def cmd_view_environments(cfg: Config):
    """
    View available experiment training environments.
    """
    from hafnia.platform import get_environments, pretty_print_training_environments

    envs = get_environments(cfg.get_platform_endpoint("experiment_environments"), cfg.api_key)

    pretty_print_training_environments(envs)


def default_experiment_run_name():
    return f"run-{utils.now_as_str()}"


@experiment.command(name="create")
@click.option(
    "-n",
    "--name",
    type=str,
    default=default_experiment_run_name(),
    required=False,
    help=f"Name of the experiment. [default: run-[DATETIME] e.g. {default_experiment_run_name()}] ",
)
@click.option(
    "-c",
    "--cmd",
    type=str,
    default="python scripts/train.py",
    show_default=True,
    help="Command to run the experiment.",
)
@click.option(
    "-p",
    "--trainer-path",
    type=Path,
    default=None,
    help="Path to the trainer package directory. ",
)
@click.option(
    "-i",
    "--trainer-id",
    type=str,
    default=None,
    help="ID of the trainer package. View available trainers with 'hafnia trainer ls'",
)
@click.option(
    "-d",
    "--dataset",
    type=str,
    default=None,
    required=False,
    help="DatasetIdentifier: Name of the dataset. View Available datasets with 'hafnia dataset ls'",
)
@click.option(
    "-r",
    "--dataset-recipe",
    type=str,
    default=None,
    required=False,
    help="DatasetIdentifier: Name of the dataset recipe. View available dataset recipes with 'hafnia dataset-recipe ls'",
)
@click.option(
    "--dataset-recipe-id",
    type=str,
    default=None,
    required=False,
    help="DatasetIdentifier: ID of the dataset recipe. View dataset recipes with 'hafnia dataset-recipe ls'",
)
@click.option(
    "-e",
    "--environment",
    type=str,
    default="Free Tier",
    show_default=True,
    help="Experiment environment name. View available environments with 'hafnia experiment environments'",
)
@click.pass_obj
def cmd_create_experiment(
    cfg: Config,
    name: str,
    cmd: str,
    trainer_path: Path,
    trainer_id: Optional[str],
    dataset: Optional[str],
    dataset_recipe: Optional[str],
    dataset_recipe_id: Optional[str],
    environment: str,
) -> None:
    """
    Create and launch a new experiment run

    Requires one dataset recipe and one trainer package:.
        - One dataset identifier is required either '--dataset', '--dataset-recipe' or '--dataset-recipe-id'.
        - One trainer identifier is required either '--trainer-path' or '--trainer-id'.

    \b
    Examples:
    # Launch an experiment with a dataset and a trainer package from local path
    hafnia experiment create --dataset mnist --trainer-path ../trainer-classification

    \b
    # Launch experiment with dataset recipe by name and trainer package by id
    hafnia experiment create --dataset-recipe mnist-recipe --trainer-id 5e454c0d-fdf1-4d1f-9732-771d7fecd28e

    \b
    # Show available options:
    hafnia experiment create --name "My Experiment" -d mnist --cmd "python scripts/train.py" -e "Free Tier" -p ../trainer-classification
    """
    from hafnia.platform import create_experiment, get_exp_environment_id

    dataset_recipe_response = get_dataset_recipe_by_dataset_identifies(
        cfg=cfg,
        dataset_name=dataset,
        dataset_recipe_name=dataset_recipe,
        dataset_recipe_id=dataset_recipe_id,
    )
    dataset_recipe_id = dataset_recipe_response["id"]

    trainer_id = get_trainer_package_by_identifies(
        cfg=cfg,
        trainer_path=trainer_path,
        trainer_id=trainer_id,
    )

    env_id = get_exp_environment_id(environment, cfg.get_platform_endpoint("experiment_environments"), cfg.api_key)

    experiment = create_experiment(
        experiment_name=name,
        dataset_recipe_id=dataset_recipe_id,
        trainer_id=trainer_id,
        exec_cmd=cmd,
        environment_id=env_id,
        endpoint=cfg.get_platform_endpoint("experiments"),
        api_key=cfg.api_key,
    )

    experiment_properties = {
        "ID": experiment.get("id", "N/A"),
        "Name": experiment.get("name", "N/A"),
        "State": experiment.get("state", "N/A"),
        "Trainer Package ID": experiment.get("trainer", "N/A"),
        "Dataset Recipe ID": experiment.get("dataset_recipe", "N/A"),
        "Dataset ID": experiment.get("dataset", "N/A"),
        "Created At": experiment.get("created_at", "N/A"),
    }
    print("Successfully created experiment: ")
    for key, value in experiment_properties.items():
        print(f"  {key}: {value}")


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


def get_trainer_package_by_identifies(
    cfg: Config,
    trainer_path: Optional[Path],
    trainer_id: Optional[str],
) -> str:
    from hafnia.platform import get_trainer_package_by_id

    if trainer_path is not None and trainer_id is not None:
        raise click.ClickException(
            "Multiple trainer identifiers (--trainer-path, --trainer-id) have been provided. Define only one."
        )

    if trainer_path is not None:
        trainer_path = Path(trainer_path)
        if not trainer_path.exists():
            raise click.ClickException(f"Trainer package path '{trainer_path}' does not exist.")
        trainer_id = create_trainer_package(
            trainer_path,
            cfg.get_platform_endpoint("trainers"),
            cfg.api_key,
        )
        return trainer_id

    if trainer_id:
        trainer_response = get_trainer_package_by_id(
            id=trainer_id, endpoint=cfg.get_platform_endpoint("trainers"), api_key=cfg.api_key
        )
        return trainer_response["id"]

    raise click.MissingParameter(
        "At least one trainer identifier must be provided. Set one of the following:\n"
        "  --trainer-path <path>  -- E.g. '--trainer-path .'\n"
        "  --trainer-id <id>  -- E.g. '--trainer-id 5e454c0d-fdf1-4d1f-9732-771d7fecd28e'\n"
    )
