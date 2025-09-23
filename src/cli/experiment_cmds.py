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
from hafnia.platform.train_recipe import create_training_recipe


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
    "--train-recipe-path",
    type=Path,
    default=None,
    help="Path to the training recipe directory. ",
)
@click.option(
    "-i",
    "--train-recipe-id",
    type=str,
    default=None,
    help="ID of the training recipe. View available training recipes with 'hafnia training-recipe ls'",
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
    train_recipe_path: Path,
    train_recipe_id: Optional[str],
    dataset: Optional[str],
    dataset_recipe: Optional[str],
    dataset_recipe_id: Optional[str],
    environment: str,
) -> None:
    """
    Create and launch a new experiment run

    Requires one dataset recipe and one training recipe:.
        - One dataset identifier is required either '--dataset', '--dataset-recipe' or '--dataset-recipe-id'.
        - One training recipe identifier is required either '--train-recipe-path' or '--train-recipe-id'.

    \b
    Examples:
    # Launch an experiment with a dataset and a training recipe from local path
    hafnia experiment create --dataset mnist --train-recipe-path ../recipe-classification

    \b
    # Launch experiment with dataset recipe by name and training recipe by id
    hafnia experiment create --dataset-recipe mnist-recipe --train-recipe-id 5e454c0d-fdf1-4d1f-9732-771d7fecd28e

    \b
    # Show available options:
    hafnia experiment create --name "My Experiment" -d mnist --cmd "python scripts/train.py" -e "Free Tier" -p ../recipe-classification
    """
    from hafnia.platform import create_experiment, get_exp_environment_id

    dataset_recipe_response = get_dataset_recipe_by_dataset_identifies(
        cfg=cfg,
        dataset_name=dataset,
        dataset_recipe_name=dataset_recipe,
        dataset_recipe_id=dataset_recipe_id,
    )
    dataset_recipe_id = dataset_recipe_response["id"]

    train_recipe_id = get_training_recipe_by_identifies(
        cfg=cfg,
        train_recipe_path=train_recipe_path,
        train_recipe_id=train_recipe_id,
    )

    env_id = get_exp_environment_id(environment, cfg.get_platform_endpoint("experiment_environments"), cfg.api_key)

    experiment = create_experiment(
        experiment_name=name,
        dataset_recipe_id=dataset_recipe_id,
        training_recipe_id=train_recipe_id,
        exec_cmd=cmd,
        environment_id=env_id,
        endpoint=cfg.get_platform_endpoint("experiments"),
        api_key=cfg.api_key,
    )

    experiment_properties = {
        "ID": experiment.get("id", "N/A"),
        "Name": experiment.get("name", "N/A"),
        "State": experiment.get("state", "N/A"),
        "Training Recipe ID": experiment.get("recipe", "N/A"),
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
