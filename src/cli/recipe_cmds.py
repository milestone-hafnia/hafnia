from pathlib import Path

import click

import cli.consts as consts


@click.group(name="recipe")
def recipe() -> None:
    """Hafnia Recipe management commands"""
    pass


@recipe.command(name="create")
@click.argument("source")
@click.option(
    "--output", type=click.Path(writable=True), default="./recipe.zip", show_default=True, help="Output recipe path."
)
def create(source: str, output: str) -> None:
    """Create HRF from local path"""

    from hafnia.utils import archive_dir

    path_output_zip = Path(output)
    if path_output_zip.suffix != ".zip":
        raise click.ClickException(consts.ERROR_RECIPE_FILE_FORMAT)

    path_source = Path(source)
    path_output_zip = archive_dir(path_source, path_output_zip)


@recipe.command(name="view")
@click.option("--path", type=str, default="./recipe.zip", show_default=True, help="Path of recipe.zip.")
@click.option("--depth-limit", type=int, default=3, help="Limit the depth of the tree view.", show_default=True)
def view(path: str, depth_limit: int) -> None:
    """View the content of a recipe zip file."""
    from hafnia.utils import show_recipe_content

    path_recipe = Path(path)
    if not path_recipe.exists():
        raise click.ClickException(
            f"Recipe file '{path_recipe}' does not exist. Please provide a valid path. "
            f"To create a recipe, use the 'hafnia recipe create' command."
        )
    show_recipe_content(path_recipe, depth_limit=depth_limit)
