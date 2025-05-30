import json
import subprocess
import zipfile
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Optional

import click

from cli.config import Config


@click.group(name="runc")
def runc():
    """Experiment management commands"""
    pass


@runc.command(name="launch")
@click.argument("task", required=True)
def launch(task: str) -> None:
    """Launch a job within the image."""
    from hafnia.platform.executor import handle_launch

    handle_launch(task)


@runc.command(name="launch-local")
@click.argument("exec_cmd", type=str)
@click.option(
    "--dataset",
    type=str,
    help="Hafnia dataset name e.g. mnist, midwest-vehicle-detection or a path to a local dataset",
    required=True,
)
@click.option(
    "--image_name",
    type=Optional[str],
    default=None,
    help=(
        "Docker image name to use for the launch. "
        "By default, it will use image name from '.state.json' "
        "file generated by the 'hafnia runc build-local' command"
    ),
)
@click.pass_obj
def launch_local(cfg: Config, exec_cmd: str, dataset: str, image_name: str) -> None:
    """Launch a job within the image."""
    from hafnia.data.factory import download_or_get_dataset_path

    is_local_dataset = "/" in dataset
    if is_local_dataset:
        click.echo(f"Using local dataset: {dataset}")
        path_dataset = Path(dataset)
        if not path_dataset.exists():
            raise click.ClickException(f"Dataset path does not exist: {path_dataset}")
    else:
        click.echo(f"Using Hafnia dataset: {dataset}")
        path_dataset = download_or_get_dataset_path(dataset_name=dataset, cfg=cfg, force_redownload=False)

    if image_name is None:
        # Load image name from state.json
        path_state_file = Path("state.json")
        if not path_state_file.exists():
            raise click.ClickException("State file does not exist. Please build the image first.")
        state_dict = json.loads(path_state_file.read_text())
        if "mdi_tag" not in state_dict:
            raise click.ClickException("mdi_tag not found in state file. Please build the image first.")
        image_name = state_dict["mdi_tag"]

    docker_cmds = [
        "docker",
        "run",
        "--rm",
        "-v",
        f"{path_dataset.absolute()}:/opt/ml/input/data/training",
        "-e",
        "HAFNIA_CLOUD=true",
        "-e",
        "PYTHONPATH=src",
        "--runtime",
        "nvidia",
        image_name,
    ] + exec_cmd.split(" ")

    # Use the "hafnia runc launch" cmd when we have moved to the new folder structure and
    # direct commands.
    # Replace '+ exec_cmd.split(" ")' with '["hafnia", "runc", "launch"] + exec_cmd.split(" ")'

    click.echo(f"Running command: \n\t{' '.join(docker_cmds)}")
    subprocess.run(docker_cmds, check=True)


@runc.command(name="build")
@click.argument("recipe_url")
@click.argument("state_file", default="state.json")
@click.argument("ecr_repository", default="localhost")
@click.argument("image_name", default="recipe")
@click.pass_obj
def build(cfg: Config, recipe_url: str, state_file: str, ecr_repository: str, image_name: str) -> None:
    """Build docker image with a given recipe."""
    from hafnia.platform.builder import build_image, prepare_recipe

    with TemporaryDirectory() as temp_dir:
        image_info = prepare_recipe(recipe_url, Path(temp_dir), cfg.api_key)
        image_info["name"] = image_name
        build_image(image_info, ecr_repository, state_file)


@runc.command(name="build-local")
@click.argument("recipe")
@click.argument("state_file", default="state.json")
@click.argument("image_name", default="recipe")
def build_local(recipe: str, state_file: str, image_name: str) -> None:
    """Build recipe from local path as image with prefix - localhost"""

    from hafnia.platform.builder import build_image, validate_recipe
    from hafnia.utils import archive_dir

    recipe_zip = Path(recipe)
    recipe_created = False
    if not recipe_zip.suffix == ".zip" and recipe_zip.is_dir():
        recipe_zip = archive_dir(recipe_zip)
        recipe_created = True

    validate_recipe(recipe_zip)
    click.echo("Recipe successfully validated")
    with TemporaryDirectory() as temp_dir:
        temp_dir_path = Path(temp_dir)
        with zipfile.ZipFile(recipe_zip, "r") as zip_ref:
            zip_ref.extractall(temp_dir_path)

        image_info = {
            "name": image_name,
            "dockerfile": (temp_dir_path / "Dockerfile").as_posix(),
            "docker_context": temp_dir_path.as_posix(),
            "hash": sha256(recipe_zip.read_bytes()).hexdigest()[:8],
        }
        click.echo("Start building image")
        build_image(image_info, "localhost", state_file=state_file)
        if recipe_created:
            recipe_zip.unlink()
