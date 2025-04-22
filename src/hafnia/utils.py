import functools
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional
from zipfile import ZipFile

import click

from hafnia.log import logger

PATH_DATA = Path("./.data")
PATH_DATASET = PATH_DATA / "datasets"
PATH_RECIPES = PATH_DATA / "recipes"


def now_as_str() -> str:
    """Get the current date and time as a string."""
    return datetime.now().strftime("%Y-%m-%dT%H-%M-%S")


def get_recipe_path(recipe_name: str) -> Path:
    now = now_as_str()
    path_recipe = PATH_RECIPES / f"{recipe_name}_{now}.zip"
    return path_recipe


def archive_dir(recipe_path: Path, output_path: Optional[Path] = None) -> Path:
    recipe_zip_path = output_path or recipe_path / "recipe.zip"
    assert recipe_zip_path.suffix == ".zip", "Output path must be a zip file"
    recipe_zip_path.parent.mkdir(parents=True, exist_ok=True)

    click.echo(f"Creating zip archive {recipe_path}")
    with ZipFile(recipe_zip_path, "w") as zip_ref:
        for item in recipe_path.rglob("*"):
            should_skip = (
                item == recipe_zip_path
                or item.name.endswith(".zip")
                or any(part.startswith(".") for part in item.parts)
                or any(part == "__pycache__" for part in item.parts)
            )

            if should_skip:
                if item != recipe_zip_path:
                    click.echo(f"[-] {item.relative_to(recipe_path)}")
                continue

            if not item.is_file():
                continue

            relative_path = item.relative_to(recipe_path)
            click.echo(f"[+] {relative_path}")
            zip_ref.write(item, relative_path)
    return recipe_zip_path


def safe(func: Callable) -> Callable:
    """
    Decorator that catches exceptions, logs them, and exits with code 1.

    Args:
        func: The function to decorate

    Returns:
        Wrapped function that handles exceptions
    """

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error in {func.__name__}: {str(e)}")
            sys.exit(1)

    return wrapper


def is_remote_job() -> bool:
    """Check if the current job is running in HAFNIA cloud environment."""
    is_remote = os.getenv("HAFNIA_CLOUD", "false").lower() == "true"
    return is_remote
