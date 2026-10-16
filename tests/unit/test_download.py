from pathlib import Path
from unittest.mock import MagicMock

import pytest

from hafnia.platform.download import download_single_object, ensure_platform_origin

PLATFORM_URL = "https://api.hafnia.example"


def test_download_single_object_writes_inside_output_dir(tmp_path: Path) -> None:
    s3_client = MagicMock()
    object_key = "experiments/trainer/1/package.zip"

    local_path = download_single_object(s3_client, "bucket", object_key, tmp_path)

    assert local_path == tmp_path.resolve() / object_key
    assert local_path.parent.is_dir()
    s3_client.download_file.assert_called_once_with("bucket", object_key, local_path.as_posix())


@pytest.mark.parametrize(
    "object_key",
    [
        "/root/.pyenv/versions/3.10.20/lib/python3.10/encodings/cp437.py",
        "../outside.py",
        "nested/../../outside.py",
        "",
    ],
)
def test_download_single_object_rejects_keys_outside_output_dir(tmp_path: Path, object_key: str) -> None:
    s3_client = MagicMock()

    with pytest.raises(ValueError, match="inside the output directory"):
        download_single_object(s3_client, "bucket", object_key, tmp_path)

    s3_client.download_file.assert_not_called()
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    "url",
    [
        f"{PLATFORM_URL}/api/v1/experiments/1/trainer-credentials",
        "HTTPS://API.HAFNIA.EXAMPLE/api/v1/experiments/1/trainer-credentials",
    ],
)
def test_ensure_platform_origin_accepts_platform_urls(url: str) -> None:
    ensure_platform_origin(url, PLATFORM_URL)


@pytest.mark.parametrize(
    "url",
    [
        "https://attacker.example/api/v1/experiments/1/trainer-credentials",
        "http://api.hafnia.example/api/v1/experiments/1/trainer-credentials",
        "https://api.hafnia.example:8443/api/v1/experiments/1/trainer-credentials",
        "https://api.hafnia.example@attacker.example/api/v1/experiments/1/trainer-credentials",
        "https://api.hafnia.example.attacker.example/api/v1/experiments/1/trainer-credentials",
    ],
)
def test_ensure_platform_origin_rejects_other_origins(url: str) -> None:
    with pytest.raises(ValueError, match="not on the configured platform"):
        ensure_platform_origin(url, PLATFORM_URL)


def test_ensure_platform_origin_requires_a_configured_platform_url() -> None:
    with pytest.raises(ValueError, match="platform_url is not configured"):
        ensure_platform_origin(f"{PLATFORM_URL}/api/v1/experiments/1/trainer-credentials", "")
