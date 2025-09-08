from pathlib import Path
from typing import Optional

import click

from cli.config import Config
from hafnia import utils


@click.group(name="experiment")
def experiment() -> None:
    """Experiment management commands"""
    pass


@experiment.command(name="environments")
@click.pass_obj
def view_environments(cfg: Config):
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
    default=None,
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
    help="Path to the training recipe directory. Uses current working directory as default.",
)
@click.option(
    "-p",
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
    "-i",
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
def create(
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

    This command allows you to create a new experiment run with the specified parameters.
    Requires one dataset recipe and one training recipe to be specified.
        - One dataset identifier is required either '--dataset', '--dataset-recipe' or '--dataset-recipe-id'.
        - One training recipe identifier is required either '--train-recipe-path' or '--train-recipe-id'.

    \b
    Examples:
    # Launch an experiment with 'mnist' and the
    hafnia experiment create --dataset mnist --train-recipe-path ../recipe-classification

    \b
    # Launch experiment with dataset recipe by name and training recipe by id
    hafnia experiment create --dataset-recipe mnist-recipe --train-recipe-id 5e454c0d-fdf1-4d1f-9732-771d7fecd28e

    \b
    # Use all options
    hafnia experiment create --name "My Experiment" -d mnist --cmd "python scripts/train.py" -e "Free Tier" -p ../recipe-classification
    """
    from hafnia.platform import create_experiment, get_dataset_id, get_exp_environment_id
    from hafnia.platform.experiment import get_dataset_recipe_by_dataset_identifies, get_training_recipe_by_identifies

    if name is None:
        name = default_experiment_run_name()

    if dataset:  # TODO: Deprecated. Remove the following 4 lines when s2m and TaaS support dataset recipes
        dataset_id = get_dataset_id(dataset, cfg.get_platform_endpoint("datasets"), cfg.api_key)
    else:
        raise NotImplementedError("Dataset recipes are not supported in s2m yet. Only 'dataset_name' is supported.")

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
        dataset_id=dataset_id,
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
