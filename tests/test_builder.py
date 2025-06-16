from pathlib import Path
from unittest.mock import MagicMock
from zipfile import ZipFile

import pytest

from hafnia.platform.builder import check_registry, validate_recipe_format


@pytest.fixture
def valid_recipe(tmp_path: Path) -> Path:
    from zipfile import ZipFile

    zip_path = tmp_path / "valid_recipe.zip"
    with ZipFile(zip_path, "w") as zipf:
        zipf.writestr("src/lib/example.py", "# Example lib")
        zipf.writestr("scripts/run.py", "print('Running training.')")
        zipf.writestr("Dockerfile", "FROM python:3.9")
    return zip_path


def test_valid_recipe_structure(valid_recipe: Path) -> None:
    """Test validation with a correctly structured zip file."""
    validate_recipe_format(valid_recipe)


def test_validate_recipe_no_scripts(tmp_path: Path) -> None:
    """Test validation fails when no Python scripts are present."""
    from zipfile import ZipFile

    zip_path = tmp_path / "no_scripts.zip"
    with ZipFile(zip_path, "w") as zipf:
        zipf.writestr("src/lib/example.py", "# Example lib")
        zipf.writestr("Dockerfile", "FROM python:3.9")

    with pytest.raises(FileNotFoundError) as excinfo:
        validate_recipe_format(zip_path)

    assert "Wrong recipe structure" in str(excinfo.value)


def test_invalid_recipe_structure(tmp_path: Path) -> None:
    """Test validation with an incorrectly structured zip file."""
    zip_path = tmp_path / "invalid_recipe.zip"
    with ZipFile(zip_path, "w") as zipf:
        zipf.writestr("README.md", "# Example readme")

    with pytest.raises(FileNotFoundError) as excinfo:
        validate_recipe_format(zip_path)

    error_msg = str(excinfo.value)
    assert "Wrong recipe structure" in error_msg


def test_successful_recipe_extraction(valid_recipe: Path, tmp_path: Path) -> None:
    """Test successful recipe download and extraction."""

    from hashlib import sha256

    from hafnia.platform.builder import prepare_recipe

    state_file = "state.json"
    expected_hash = sha256(valid_recipe.read_bytes()).hexdigest()[:8]

    with pytest.MonkeyPatch.context() as mp:
        mock_download = MagicMock(return_value={"status": "success", "downloaded_files": [valid_recipe]})

        mp.setattr("hafnia.platform.builder.download_resource", mock_download)
        result = prepare_recipe("s3://bucket/recipe.zip", tmp_path, "api-key-123", Path(state_file))
        mock_download.assert_called_once_with("s3://bucket/recipe.zip", tmp_path.as_posix(), "api-key-123")

        assert result["digest"] == expected_hash


def test_ecr_image_exist() -> None:
    """Test when image exists in ECR."""
    mock_ecr_client = MagicMock()
    mock_ecr_client.describe_images.return_value = {"imageDetails": [{"imageTags": ["v1.0"], "imageDigest": "1234a"}]}

    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("AWS_REGION", "us-west-1")
        mp.setattr("boto3.client", lambda service, **kwargs: mock_ecr_client)
        result = check_registry("my-repo:v1.0")
        assert result == "1234a"


def test_ecr_image_not_found() -> None:
    """Test when ECR client raises ImageNotFoundException."""

    from botocore.exceptions import ClientError

    mock_ecr_client = MagicMock()
    mock_ecr_client.describe_images.side_effect = ClientError(
        {"Error": {"Code": "ImageNotFoundException"}}, "describe_images"
    )

    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("AWS_REGION", "us-west-2")
        mp.setattr("boto3.client", lambda service, **kwargs: mock_ecr_client)
        result = check_registry("my-repo:v1.0")
        assert result is None
