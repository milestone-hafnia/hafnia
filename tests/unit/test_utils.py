from pathlib import Path
from typing import Tuple
from zipfile import ZipFile

import pytest

from hafnia.utils import (
    BLOCKED_TRAINER_PACKAGE_PATTERNS,
    DEFAULT_IGNORE_SPECIFICATION,
    FILENAME_HAFNIAIGNORE,
    create_trainer_package_zip_file,
)


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


def test_zip_trainer_package_no_ignore_hafnia_file(tmp_path: Path, project_with_files_default: Tuple) -> None:
    """Test zipping a trainer package using the default ignore specification."""
    path_source_code, add_files, _ = project_with_files_default
    path_zipped_trainer = tmp_path / "trainer.zip"
    path_zipped_trainer, _ = create_trainer_package_zip_file(path_source_code, path_zipped_trainer)
    zipped_files = ZipFile(path_zipped_trainer).namelist()
    assert set(zipped_files) == set(add_files)


def test_zip_trainer_package_empty_ignore_hafnia_file(tmp_path: Path, project_with_files_default) -> None:
    """Test zipping a trainer package using a custom ignore specification."""
    path_source_code, keep_files_, ignore_files = project_with_files_default
    keep_files = keep_files_ + ignore_files

    # Create an empty .hafniaignore file to include all files
    path_ignore_file = tmp_path / FILENAME_HAFNIAIGNORE
    path_ignore_file.write_text("")

    # Automatically picks up the '.hafniaignore' file from the root of the source code
    path_zipped_trainer = tmp_path / "trainer.zip"
    # Files matching blocked patterns (e.g. '.venv') are included, so the blocked files check must be bypassed
    path_zipped_trainer, _ = create_trainer_package_zip_file(
        path_source_code, path_zipped_trainer, path_ignore_file=path_ignore_file, allow_blocked_files=True
    )

    zipped_files = ZipFile(path_zipped_trainer).namelist()
    assert set(zipped_files) == set(keep_files + ignore_files)


def test_zip_trainer_package_raises_on_blocked_files(tmp_path: Path, project_with_files_default) -> None:
    """Files matching blocked patterns must raise an error unless explicitly allowed."""
    path_source_code, _, _ = project_with_files_default
    (path_source_code / "tests" / "test_train.py").parent.mkdir(parents=True)
    (path_source_code / "tests" / "test_train.py").write_text("Some content")

    # '.hafniaignore' that ignores '.venv' but not 'tests' or '__pycache__'
    (path_source_code / FILENAME_HAFNIAIGNORE).write_text(".venv\n")

    path_zipped_trainer = tmp_path / "trainer.zip"
    with pytest.raises(ValueError, match="contains files that should not be part of a trainer package") as exc_info:
        create_trainer_package_zip_file(path_source_code, path_zipped_trainer)
    error_msg = str(exc_info.value)
    assert "'tests/'" in error_msg
    assert "'__pycache__/'" in error_msg
    assert "'.venv/'" not in error_msg
    assert FILENAME_HAFNIAIGNORE in error_msg
    assert not path_zipped_trainer.exists(), "Zip file should not be created when blocked files are found"

    path_zipped_trainer, _ = create_trainer_package_zip_file(
        path_source_code, path_zipped_trainer, allow_blocked_files=True
    )
    zipped_files = ZipFile(path_zipped_trainer).namelist()
    assert "tests/test_train.py" in zipped_files


def test_default_ignore_specification_contains_blocked_patterns() -> None:
    missing_patterns = set(BLOCKED_TRAINER_PACKAGE_PATTERNS) - set(DEFAULT_IGNORE_SPECIFICATION)
    assert not missing_patterns, f"Blocked patterns missing from default ignore specification: {missing_patterns}"


def test_zip_trainer_package_custom_ignore_hafnia_file(tmp_path: Path, project_with_files_default) -> None:
    """Test zipping a trainer package using a custom ignore specification."""

    path_source_code, keep_files, ignore_files = project_with_files_default
    all_files = keep_files + ignore_files

    # Create a .hafniaignore file that ignores all files/folders starting with '.'
    # (e.g., .venv, .git, etc.) and '__pycache__' (blocked pattern)
    ignore_patterns = [".*", "__pycache__"]
    expected_in_trainer_files = [file for file in all_files if not file.startswith((".", "__pycache__"))]

    # Place the ignore file in the source code root directory
    path_ignore_file1 = path_source_code / FILENAME_HAFNIAIGNORE
    path_ignore_file1.write_text("\n".join(ignore_patterns))

    # Automatically picks up the '.hafniaignore' file from the root of the source code
    path_zipped_trainer1 = tmp_path / "trainer.zip"
    path_zipped_trainer1, _ = create_trainer_package_zip_file(path_source_code, path_zipped_trainer1)
    zipped_files1 = ZipFile(path_zipped_trainer1).namelist()
    assert set(expected_in_trainer_files) == set(zipped_files1)


_title_to_name_cases = [
    ("My Dataset", "my_dataset"),
    ("My-Dataset", "my-dataset"),
    ("Dataset Title !*'\n", "dataset_title"),
    ("   Leading and trailing spaces   ", "leading_and_trailing_spaces"),
    ("Multiple   Spaces", "multiple_spaces"),
    ("Special_Characters@#$%^&*()!", "special_characters"),
    ("Mixed-CASE Title", "mixed-case_title"),
]


@pytest.mark.parametrize(
    "title, expected_name",
    _title_to_name_cases,
    ids=[f"{tc[0]} -> {tc[1]}" for tc in _title_to_name_cases],
)
def test_title_to_name(title: str, expected_name: str) -> None:
    from hafnia.utils import title_to_name

    assert title_to_name(title) == expected_name


_name_to_title_cases = [
    ("my_dataset", "My Dataset"),
    ("my-dataset", "My Dataset"),
    ("dataset_title", "Dataset Title"),
    ("my__dataset__v2", "My  Dataset  V2"),
    ("mixed-case_title", "Mixed Case Title"),
    ("single", "Single"),
]


@pytest.mark.parametrize(
    "name, expected_title",
    _name_to_title_cases,
    ids=[f"{tc[0]} -> {tc[1]}" for tc in _name_to_title_cases],
)
def test_name_to_title(name: str, expected_title: str) -> None:
    from hafnia.utils import name_to_title

    assert name_to_title(name) == expected_title


def test_zip_entry_names_use_forward_slashes(tmp_path: Path, project_with_files_default: Tuple) -> None:
    """Zip entry names must use forward slashes so Linux can extract the directory structure correctly.
    On Windows, pathspec.match_tree() returns backslash paths -- this test catches that regression."""
    path_source_code, _, _ = project_with_files_default
    path_zipped_trainer = tmp_path / "trainer.zip"
    path_zipped_trainer, _ = create_trainer_package_zip_file(path_source_code, path_zipped_trainer)
    zipped_files = ZipFile(path_zipped_trainer).namelist()
    backslash_entries = [e for e in zipped_files if "\\" in e]
    assert not backslash_entries, f"Backslash found in zip entry names: {backslash_entries}"
