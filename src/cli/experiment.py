from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import click

from mdi_python_tools.log import logger
from mdi_python_tools.platform.codebuild import validate_recipe


@click.group(name="experiment")
def experiment_group() -> None:
    """System management commands"""
    pass


@experiment_group.command(name="create")
@click.argument("source_dir")
def create(source_dir: str) -> None:
    source_dir = Path(source_dir)
    if not source_dir.exists():
        raise FileNotFoundError(f"Source directory {source_dir} does not exist.")

    with TemporaryDirectory() as tdir:
        zip_path = Path(tdir) / "recipe.zip"
        with ZipFile(zip_path, "w") as archive:
            for path in source_dir.rglob("*"):
                if any(
                    part.startswith(".") or part.startswith("__")
                    for part in path.relative_to(source_dir).parts
                ):
                    continue
                archive.write(path, path.relative_to(source_dir))
        validate_recipe(zip_path)
        logger.info("Recipe validated successfully.")
