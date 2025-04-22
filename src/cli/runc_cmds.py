from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory

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
    image_info = {
        "name": image_name,
        "dockerfile": f"{recipe_zip.parent}/Dockerfile",
        "docker_context": f"{recipe_zip.parent}",
        "hash": sha256(recipe_zip.read_bytes()).hexdigest()[:8],
    }
    click.echo("Start building image")
    build_image(image_info, "localhost", state_file=state_file)
    if recipe_created:
        recipe_zip.unlink()
