from pathlib import Path
from unittest.mock import MagicMock

import pytest

from hafnia.platform.builder import check_ecr, validate_recipe


@pytest.fixture
def valid_recipe(tmp_path: Path) -> Path:
    from zipfile import ZipFile

    zip_path = tmp_path / "valid_recipe.zip"
    with ZipFile(zip_path, "w") as zipf:
        zipf.writestr("src/lib/example.py", "# Example lib")
        zipf.writestr("src/scripts/run.py", "print('Running training.')")
        zipf.writestr("Dockerfile", "FROM python:3.9")
    return zip_path


@pytest.fixture(scope="function")
def mock_boto_session() -> MagicMock:
    mock_client = MagicMock()
    mock_session = MagicMock()
    mock_session.client.return_value = mock_client
    return mock_client


def test_valid_recipe_structure(valid_recipe: Path) -> None:
    """Test validation with a correctly structured zip file."""
    validate_recipe(valid_recipe)


def test_validate_recipe_no_scripts(tmp_path: Path) -> None:
    """Test validation fails when no Python scripts are present."""
    from zipfile import ZipFile

    zip_path = tmp_path / "no_scripts.zip"
    with ZipFile(zip_path, "w") as zipf:
        zipf.writestr("src/lib/example.py", "# Example lib")
        zipf.writestr("src/scripts/README.md", "# Not a Python file")
        zipf.writestr("Dockerfile", "FROM python:3.9")

    with pytest.raises(ValueError) as excinfo:
        validate_recipe(zip_path)

    assert "No Python script files found in the 'src/scripts/' directory." in str(excinfo.value)


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
    for required_path in ("Dockerfile", "src/lib/", "src/scripts/"):
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
