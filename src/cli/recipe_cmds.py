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

    from hafnia.platform.builder import validate_recipe
    from hafnia.utils import archive_dir

    path_output_zip = Path(output)
    if path_output_zip.suffix != ".zip":
        raise click.ClickException(consts.ERROR_RECIPE_FILE_FORMAT)

    path_source = Path(source)
    path_output_zip = archive_dir(path_source, path_output_zip)
    validate_recipe(path_output_zip)
