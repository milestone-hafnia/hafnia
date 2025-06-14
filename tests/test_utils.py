from pathlib import Path
from typing import Tuple
from zipfile import ZipFile

import pytest

from hafnia.utils import FILENAME_HAFNIAIGNORE, archive_dir


@pytest.fixture
def project_with_files_default(tmp_path: Path) -> Tuple[Path, list[str], list[str]]:
    zip_files = [
        "scripts/train.py",
        "scripts/README.md",
        "Dockerfile",
        "src/lib/example.py",
    ]

    ignore_files = [
        ".venv/bin/activate",
        ".venv/lib/jedi/__init__.py",
        "__pycache__/some_file.py",
        "__pycache__/example.cpython-310.pyc",
    ]
    files = [*zip_files, *ignore_files]
    path_source_code = tmp_path / "source_code"

    for f in files:
        is_folder = f.endswith("/")
        path = path_source_code / f
        if is_folder:
            path.mkdir(parents=True, exist_ok=True)
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("Some content")
    return path_source_code, zip_files, ignore_files


def test_zip_recipe_no_ignore_hafnia_file(tmp_path: Path, project_with_files_default: Tuple) -> None:
    """Test zipping a recipe using the default ignore specification."""
    path_source_code, add_files, _ = project_with_files_default
    path_zipped_recipe = tmp_path / "recipe.zip"
    path_zipped_recipe = archive_dir(path_source_code, path_zipped_recipe)
    zipped_files = ZipFile(path_zipped_recipe).namelist()
    assert set(zipped_files) == set(add_files)


def test_zip_recipe_empty_ignore_hafnia_file(tmp_path: Path, project_with_files_default) -> None:
    """Test zipping a recipe using a custom ignore specification."""
    path_source_code, keep_files_, ignore_files = project_with_files_default
    keep_files = keep_files_ + ignore_files

    # Create an empty .hafniaignore file to include all files
    path_ignore_file = tmp_path / FILENAME_HAFNIAIGNORE
    path_ignore_file.write_text("")

    # Automatically picks up the '.hafniaignore' file from the root of the source code
    path_zipped_recipe = tmp_path / "recipe.zip"
    path_zipped_recipe = archive_dir(path_source_code, path_zipped_recipe, path_ignore_file=path_ignore_file)

    zipped_files = ZipFile(path_zipped_recipe).namelist()
    assert set(zipped_files) == set(keep_files + ignore_files)


def test_zip_recipe_custom_ignore_hafnia_file(tmp_path: Path, project_with_files_default) -> None:
    """Test zipping a recipe using a custom ignore specification."""

    path_source_code, keep_files, ignore_files = project_with_files_default
    all_files = keep_files + ignore_files

    # Create a .hafniaignore file that ignores all files/folders starting with '.'
    # (e.g., .venv, .git, etc.)
    ignore_patterns = [".*"]
    expected_in_recipe_files = [file for file in all_files if not file.startswith(".")]

    # Place the ignore file in the source code root directory
    path_ignore_file1 = path_source_code / FILENAME_HAFNIAIGNORE
    path_ignore_file1.write_text("\n".join(ignore_patterns))

    # Automatically picks up the '.hafniaignore' file from the root of the source code
    path_zipped_recipe1 = tmp_path / "recipe.zip"
    path_zipped_recipe1 = archive_dir(path_source_code, path_zipped_recipe1)
    zipped_files1 = ZipFile(path_zipped_recipe1).namelist()
    assert set(expected_in_recipe_files) == set(zipped_files1)
