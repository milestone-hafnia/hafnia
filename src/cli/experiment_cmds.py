from pathlib import Path

import click
from rich import print as rprint

import cli.consts as consts
from cli.config import Config


@click.group(name="experiment")
def experiment() -> None:
    """Experiment management commands"""
    pass


@experiment.command(name="create")
@click.argument("name")
@click.argument("source_dir", type=Path)
@click.argument("exec_cmd", type=str)
@click.argument("dataset_name")
@click.argument("env_name")
@click.pass_obj
def create(cfg: Config, name: str, source_dir: Path, exec_cmd: str, dataset_name: str, env_name: str) -> None:
    """Create a new experiment run"""
    from hafnia.platform import create_experiment, create_recipe, get_dataset_id, get_exp_environment_id

    if not source_dir.exists():
        raise click.ClickException(consts.ERROR_EXPERIMENT_DIR)

    try:
        dataset_id = get_dataset_id(dataset_name, cfg.get_platform_endpoint("datasets"), cfg.api_key)
    except Exception:
        raise click.ClickException(f"Error retrieving dataset '{dataset_name}'.")

    try:
        recipe_id = create_recipe(source_dir, cfg.get_platform_endpoint("recipes"), cfg.api_key)
    except Exception:
        raise click.ClickException(f"Failed to create recipe from '{source_dir}'")

    try:
        env_id = get_exp_environment_id(env_name, cfg.get_platform_endpoint("experiment_environments"), cfg.api_key)
    except Exception:
        raise click.ClickException(f"Environment '{env_name}' not found")

    try:
        experiment_id = create_experiment(
            name, dataset_id, recipe_id, exec_cmd, env_id, cfg.get_platform_endpoint("experiments"), cfg.api_key
        )
    except Exception:
        raise click.ClickException(f"Failed to create experiment '{name}'")

    rprint(
        {
            "dataset_id": dataset_id,
            "recipe_id": recipe_id,
            "environment_id": env_id,
            "experiment_id": experiment_id,
            "status": "CREATED",
        }
    )
