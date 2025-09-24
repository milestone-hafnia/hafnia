from pathlib import Path
from typing import Optional

import click

import cli.consts as consts
from cli.config import Config


@click.group(name="train-recipe")
def training_recipe() -> None:
    """Training recipe commands"""
    pass


@training_recipe.command(name="ls")
@click.pass_obj
@click.option("-l", "--limit", type=int, default=None, help="Limit number of listed dataset recipes.")
def cmd_list_training_recipes(cfg: Config, limit: Optional[int]) -> None:
    """List available training recipes on the platform"""

    from hafnia.platform.train_recipe import get_training_recipes, pretty_print_training_recipes

    endpoint = cfg.get_platform_endpoint("trainers")
    recipes = get_training_recipes(endpoint, cfg.api_key)

    pretty_print_training_recipes(recipes, limit=limit)


@training_recipe.command(name="create-zip")
@click.argument("source")
@click.option(
    "--output", type=click.Path(writable=True), default="./recipe.zip", show_default=True, help="Output recipe path."
)
def cmd_create_training_zip(source: str, output: str) -> None:
    """Create Hafnia training recipe zip-file from local path"""

    from hafnia.utils import archive_dir

    path_output_zip = Path(output)
    if path_output_zip.suffix != ".zip":
        raise click.ClickException(consts.ERROR_RECIPE_FILE_FORMAT)

    path_source = Path(source)
    path_output_zip = archive_dir(path_source, path_output_zip)


@training_recipe.command(name="view-zip")
@click.option("--path", type=str, default="./recipe.zip", show_default=True, help="Path of recipe.zip.")
@click.option("--depth-limit", type=int, default=3, help="Limit the depth of the tree view.", show_default=True)
def cmd_view_training_zip(path: str, depth_limit: int) -> None:
    """View the content of a training recipe zip file."""
    from hafnia.utils import show_recipe_content

    path_recipe = Path(path)
    if not path_recipe.exists():
        raise click.ClickException(
            f"Recipe file '{path_recipe}' does not exist. Please provide a valid path. "
            f"To create a recipe, use the 'hafnia recipe training create-zip' command."
        )
    show_recipe_content(path_recipe, depth_limit=depth_limit)
