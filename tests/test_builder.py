from pathlib import Path
from unittest.mock import MagicMock
from zipfile import ZipFile

import pytest

from hafnia.platform.builder import check_ecr, validate_recipe
from hafnia.utils import FILENAME_HAFNIAIGNORE, archive_dir


@pytest.fixture
def valid_recipe(tmp_path: Path) -> Path:
    from zipfile import ZipFile

    zip_path = tmp_path / "valid_recipe.zip"
    with ZipFile(zip_path, "w") as zipf:
        zipf.writestr("src/lib/example.py", "# Example lib")
        zipf.writestr("scripts/run.py", "print('Running training.')")
        zipf.writestr("Dockerfile", "FROM python:3.9")
    return zip_path


@pytest.fixture(scope="function")
def mock_boto_session() -> MagicMock:
    mock_client = MagicMock()
    mock_session = MagicMock()
    mock_session.client.return_value = mock_client
    return mock_client


@pytest.fixture
def project_with_files_default(tmp_path: Path) -> tuple[Path, list[str], list[str]]:
    zip_files = [
        "src/scripts/train.py",
        "src/scripts/README.md",
        "Dockerfile",
        "src/lib/example.py",
    ]

    ignore_files = [
        ".venv/bin/activate",
        ".venv/lib/jedi/__init__.py",
        "src/lib/__pycache__/some_file.py",
        "src/lib/__pycache__/example.cpython-310.pyc",
    ]
    files = [*zip_files, *ignore_files]
    path_source_code = tmp_path / "source_code"

    for file in files:
        is_folder = file.endswith("/")
        path = path_source_code / file
        if is_folder:
            path.mkdir(parents=True, exist_ok=True)
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("Some content")
    return path_source_code, zip_files, ignore_files


def test_valid_recipe_structure(valid_recipe: Path) -> None:
    """Test validation with a correctly structured zip file."""
    validate_recipe(valid_recipe)


def test_validate_recipe_no_scripts(tmp_path: Path) -> None:
    """Test validation fails when no Python scripts are present."""
    from zipfile import ZipFile

    zip_path = tmp_path / "no_scripts.zip"
    with ZipFile(zip_path, "w") as zipf:
        zipf.writestr("src/lib/example.py", "# Example lib")
        zipf.writestr("scripts/README.md", "# Not a Python file")
        zipf.writestr("Dockerfile", "FROM python:3.9")

    with pytest.raises(ValueError) as excinfo:
        validate_recipe(zip_path)

    assert "No Python script files found in the 'scripts' directory." in str(excinfo.value)


def test_zip_recipe_no_ignore_hafnia_file(tmp_path: Path, project_with_files_default) -> None:
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


def test_invalid_recipe_structure(tmp_path: Path) -> None:
    """Test validation with an incorrectly structured zip file."""
    from zipfile import ZipFile

    zip_path = tmp_path / "invalid_recipe.zip"
    with ZipFile(zip_path, "w") as zipf:
        zipf.writestr("README.md", "# Example readme")

    with pytest.raises(FileNotFoundError) as excinfo:
        validate_recipe(zip_path)

    error_msg = str(excinfo.value)
    assert "missing in the zip archive" in error_msg
    for required_path in ("Dockerfile", "src", "scripts"):
        assert required_path in error_msg


def test_successful_recipe_extraction(valid_recipe: Path, tmp_path: Path) -> None:
    """Test successful recipe download and extraction."""

    from hashlib import sha256

    from hafnia.platform.builder import get_recipe_content

    state_file = "state.json"
    expected_hash = sha256(valid_recipe.read_bytes()).hexdigest()[:8]

    with pytest.MonkeyPatch.context() as mp:
        mock_download = MagicMock(return_value={"status": "success", "downloaded_files": [valid_recipe]})
        mock_clean_up = MagicMock()

        mp.setattr("hafnia.platform.builder.download_resource", mock_download)
        mp.setattr("hafnia.platform.builder.clean_up", mock_clean_up)

        result = get_recipe_content("s3://bucket/recipe.zip", tmp_path, state_file, "api-key-123")
        mock_download.assert_called_once_with("s3://bucket/recipe.zip", tmp_path, "api-key-123")

        assert result["docker_tag"] == f"runtime:{expected_hash}"
        assert result["hash"] == expected_hash
        assert "valid_commands" in result
        assert "run" == result["valid_commands"][0]
        mock_clean_up.assert_called_once()


def test_ecr_image_exist(mock_boto_session: MagicMock) -> None:
    """Test when image exists in ECR."""

    mock_boto_session.client.return_value.describe_images.return_value = {"imageDetails": [{"imageTags": ["v1.0"]}]}
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("AWS_REGION", "us-west-2")
        mp.setattr("boto3.Session", lambda **kwargs: mock_boto_session)
        result = check_ecr("my-repo", "v1.0")
        assert result is True


def test_ecr_image_not_found(mock_boto_session: MagicMock) -> None:
    """Test when ECR client raises ImageNotFoundException."""

    from botocore.exceptions import ClientError

    mock_boto_session.client.return_value.describe_images.side_effect = ClientError(
        {"Error": {"Code": "ImageNotFoundException"}}, "describe_images"
    )

    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("AWS_REGION", "us-west-2")
        mp.setattr("boto3.Session", lambda **kwargs: mock_boto_session)
        result = check_ecr("my-repo", "v1.0")
        assert result is False
