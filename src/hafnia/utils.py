import os
import time
import zipfile
from datetime import datetime
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Iterator, Optional
from zipfile import ZipFile

import pathspec
import seedir
from rich import print as rprint

from hafnia.log import sys_logger, user_logger

PATH_DATA = Path("./.data")
PATH_DATASETS = PATH_DATA / "datasets"
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


def timed(label: str):
    """
    Decorator factory that allows custom labels for timing.
    Usage: @timed("Custom Operation")
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            operation_label = label or func.__name__
            tik = time.perf_counter()
            try:
                return func(*args, **kwargs)
            except Exception as e:
                sys_logger.error(f"{operation_label} failed: {e}")
            finally:
                elapsed = time.perf_counter() - tik
                sys_logger.debug(f"{operation_label} took {elapsed:.2f} seconds.")

        return wrapper

    return decorator


def now_as_str() -> str:
    """Get the current date and time as a string."""
    return datetime.now().strftime("%Y-%m-%dT%H-%M-%S")


def get_recipe_path(recipe_name: str) -> Path:
    now = now_as_str()
    path_recipe = PATH_RECIPES / f"{recipe_name}_{now}.zip"
    return path_recipe


def filter_recipe_files(recipe_path: Path, path_ignore_file: Optional[Path] = None) -> Iterator:
    path_ignore_file = path_ignore_file or recipe_path / FILENAME_HAFNIAIGNORE
    if not path_ignore_file.exists():
        ignore_specification_lines = DEFAULT_IGNORE_SPECIFICATION
        user_logger.info(
            f"No '{FILENAME_HAFNIAIGNORE}' was file found. Files are excluded using the default ignore patterns.\n"
            f"\tDefault ignore patterns: {DEFAULT_IGNORE_SPECIFICATION}\n"
            f"Add a '{FILENAME_HAFNIAIGNORE}' file to the root folder to make custom ignore patterns."
        )
    else:
        ignore_specification_lines = Path(path_ignore_file).read_text().splitlines()
    ignore_specification = pathspec.GitIgnoreSpec.from_lines(ignore_specification_lines)
    include_files = ignore_specification.match_tree(recipe_path, negate=True)
    return include_files


@timed("Wrapping recipe.")
def archive_dir(
    recipe_path: Path,
    output_path: Optional[Path] = None,
    path_ignore_file: Optional[Path] = None,
) -> Path:
    recipe_zip_path = output_path or recipe_path / "recipe.zip"
    assert recipe_zip_path.suffix == ".zip", "Output path must be a zip file"
    recipe_zip_path.parent.mkdir(parents=True, exist_ok=True)

    user_logger.info(f" Creating zip archive of '{recipe_path}'")
    include_files = filter_recipe_files(recipe_path, path_ignore_file)
    with ZipFile(recipe_zip_path, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as zip_ref:
        for str_filepath in include_files:
            full_path = recipe_path / str_filepath
            zip_ref.write(full_path, str_filepath)
    show_recipe_content(recipe_zip_path)

    return recipe_zip_path


def size_human_readable(size_bytes: int, suffix="B") -> str:
    size_value = float(size_bytes)
    for unit in ("", "Ki", "Mi", "Gi", "Ti", "Pi", "Ei", "Zi"):
        if abs(size_value) < 1024.0:
            return f"{size_value:3.1f} {unit}{suffix}"
        size_value /= 1024.0
    return f"{size_value:.1f}Yi{suffix}"


def show_recipe_content(recipe_path: Path, style: str = "emoji", depth_limit: int = 3) -> None:
    def scan(parent: seedir.FakeDir, path: zipfile.Path, depth: int = 0) -> None:
        if depth >= depth_limit:
            return
        for child in path.iterdir():
            if child.is_dir():
                folder = seedir.FakeDir(child.name)
                scan(folder, child, depth + 1)
                folder.parent = parent
            else:
                parent.create_file(child.name)

    recipe = seedir.FakeDir("recipe")
    scan(recipe, zipfile.Path(recipe_path))
    rprint(recipe.seedir(sort=True, first="folders", style=style, printout=False))
    user_logger.info(f"Recipe size: {size_human_readable(os.path.getsize(recipe_path))}. Max size 800 MiB")


def is_remote_job() -> bool:
    """Check if the current job is running in HAFNIA cloud environment."""
    return os.getenv("HAFNIA_CLOUD", "false").lower() == "true"
