import functools
import sys
from pathlib import Path
from typing import Any, Callable
from zipfile import ZipFile

import click

from mdi_python_tools.log import logger


def archive_dir(recipe_path: Path) -> Path:
    recipe_zip = recipe_path / "recipe.zip"
    click.echo(f"Creating zip archive {recipe_path}")
    with ZipFile(recipe_zip, "w") as zip_ref:
        for item in recipe_path.rglob("*"):
            should_skip = (
                item == recipe_zip
                or item.name.endswith(".zip")
                or any(part.startswith(".") for part in item.parts)
                or any(part.startswith("__") for part in item.parts)
            )

            if should_skip:
                if item != recipe_zip:
                    click.echo(f"[-] {item.relative_to(recipe_path)}")
                continue

            if not item.is_file():
                continue

            relative_path = item.relative_to(recipe_path)
            click.echo(f"[+] {relative_path}")
            zip_ref.write(item, relative_path)
    return recipe_zip


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
