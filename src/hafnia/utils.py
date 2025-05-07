import functools
import os
import sys
import tempfile
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional
from zipfile import ZipFile

import click
import pathspec
import seedir

from hafnia.log import logger

PATH_DATA = Path("./.data")
PATH_DATASET = PATH_DATA / "datasets"
PATH_RECIPES = PATH_DATA / "recipes"
FILENAME_HAFNIAIGNORE = ".hafniaignore"
DEFAULT_IGNORE_SPECIFICATION = [
    "*.jpg",
    "*.png",
    "*.py[cod]",
    "*_cache/",
    ".data",
    ".git",
    ".venv",
    ".vscode",
    "__pycache__",
    "recipe.zip",
    "tests",
    "wandb",
]


def now_as_str() -> str:
    """Get the current date and time as a string."""
    return datetime.now().strftime("%Y-%m-%dT%H-%M-%S")


def get_recipe_path(recipe_name: str) -> Path:
    now = now_as_str()
    path_recipe = PATH_RECIPES / f"{recipe_name}_{now}.zip"
    return path_recipe


def archive_dir(
    recipe_path: Path,
    output_path: Optional[Path] = None,
    path_ignore_file: Optional[Path] = None,
) -> Path:
    recipe_zip_path = output_path or recipe_path / "recipe.zip"
    assert recipe_zip_path.suffix == ".zip", "Output path must be a zip file"
    recipe_zip_path.parent.mkdir(parents=True, exist_ok=True)

    path_ignore_file = path_ignore_file or recipe_path / FILENAME_HAFNIAIGNORE
    if not path_ignore_file.exists():
        ignore_specification_lines = DEFAULT_IGNORE_SPECIFICATION
        click.echo(
            f"No '{FILENAME_HAFNIAIGNORE}' was file found. Files are excluded using the default ignore patterns.\n"
            f"\tDefault ignore patterns: {DEFAULT_IGNORE_SPECIFICATION}\n"
            f"Add a '{FILENAME_HAFNIAIGNORE}' file to the root folder to make custom ignore patterns."
        )
    else:
        ignore_specification_lines = Path(path_ignore_file).read_text().splitlines()
    ignore_specification = pathspec.GitIgnoreSpec.from_lines(ignore_specification_lines)

    include_files = sorted(ignore_specification.match_tree(recipe_path, negate=True))
    click.echo(f"Creating zip archive of '{recipe_path}'")
    with ZipFile(recipe_zip_path, "w") as zip_ref:
        for str_filepath in include_files:
            path_file = recipe_path / str_filepath
            if not path_file.is_file():
                continue

            relative_path = path_file.relative_to(recipe_path)
            zip_ref.write(path_file, relative_path)

    recipe_dir_tree = view_recipe_content(recipe_zip_path)
    click.echo(recipe_dir_tree)
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


def size_human_readable(size_bytes: int, suffix="B") -> str:
    # From: https://stackoverflow.com/a/1094933
    size_value = float(size_bytes)
    for unit in ("", "Ki", "Mi", "Gi", "Ti", "Pi", "Ei", "Zi"):
        if abs(size_value) < 1024.0:
            return f"{size_value:3.1f}{unit}{suffix}"
        size_value /= 1024.0
    return f"{size_value:.1f}Yi{suffix}"


def view_recipe_content(recipe_path: Path, style: str = "emoji", depth_limit: int = 3) -> str:
    zf = zipfile.ZipFile(recipe_path)
    with tempfile.TemporaryDirectory() as tempdir:
        path_extract_folder = Path(tempdir) / "recipe"
        zf.extractall(path_extract_folder)
        dir_str = seedir.seedir(
            path_extract_folder, sort=True, first="folders", style=style, depthlimit=depth_limit, printout=False
        )

    size_str = size_human_readable(os.path.getsize(recipe_path))

    dir_str = dir_str + f"\n\nRecipe size: {size_str}. Max size 800MiB\n"
    return dir_str


def is_remote_job() -> bool:
    """Check if the current job is running in HAFNIA cloud environment."""
    is_remote = os.getenv("HAFNIA_CLOUD", "false").lower() == "true"
    return is_remote
