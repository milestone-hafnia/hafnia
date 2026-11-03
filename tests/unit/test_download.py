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
    "url, platform_url",
    [
        ("https://api.hafnia.example:443/api/v1/experiments/1/trainer-credentials", "https://api.hafnia.example"),
        ("https://api.hafnia.example/api/v1/experiments/1/trainer-credentials", "https://api.hafnia.example:443"),
        ("http://localhost:80/api/v1/experiments/1/trainer-credentials", "http://localhost"),
        ("http://localhost/api/v1/experiments/1/trainer-credentials", "http://localhost:80"),
    ],
)
def test_ensure_platform_origin_treats_default_ports_as_equivalent(url: str, platform_url: str) -> None:
    ensure_platform_origin(url, platform_url)


@pytest.mark.parametrize(
    "url",
    [
        "https://attacker.example/api/v1/experiments/1/trainer-credentials",
        "http://api.hafnia.example/api/v1/experiments/1/trainer-credentials",
        "https://api.hafnia.example:8443/api/v1/experiments/1/trainer-credentials",
        "https://api.hafnia.example.attacker.example/api/v1/experiments/1/trainer-credentials",
    ],
)
def test_ensure_platform_origin_rejects_other_origins(url: str) -> None:
    with pytest.raises(ValueError, match="not on the configured platform"):
        ensure_platform_origin(url, PLATFORM_URL)


def test_ensure_platform_origin_requires_a_configured_platform_url() -> None:
    with pytest.raises(ValueError, match="platform_url is not configured"):
        ensure_platform_origin(f"{PLATFORM_URL}/api/v1/experiments/1/trainer-credentials", "")


@pytest.mark.parametrize(
    "url, platform_url",
    [
        # Scheme-less pair: urlsplit yields empty scheme/netloc for both, which a naive comparison would accept.
        ("attacker.example/api/v1/experiments/1/trainer-credentials", "api.hafnia.example"),
        ("api.hafnia.example/api/v1/experiments/1/trainer-credentials", "api.hafnia.example"),
        # Protocol-relative and non-HTTP schemes.
        ("//attacker.example/api/v1/experiments/1/trainer-credentials", PLATFORM_URL),
        ("ftp://api.hafnia.example/api/v1/experiments/1/trainer-credentials", "ftp://api.hafnia.example"),
        (f"{PLATFORM_URL}/api/v1/experiments/1/trainer-credentials", "https://"),
    ],
)
def test_ensure_platform_origin_rejects_non_absolute_http_urls(url: str, platform_url: str) -> None:
    with pytest.raises(ValueError, match="must be an absolute HTTP\\(S\\) URL"):
        ensure_platform_origin(url, platform_url)


@pytest.mark.parametrize(
    "url",
    [
        # urllib.parse.urlsplit reads the host as api.hafnia.example, but urllib3 (used by hafnia.http.fetch)
        # ends the authority at the backslash and sends the request to attacker.example.
        "https://attacker.example\\@api.hafnia.example/api/v1/experiments/1/trainer-credentials",
        "https://attacker.example\\api.hafnia.example/api/v1/experiments/1/trainer-credentials",
    ],
)
def test_ensure_platform_origin_rejects_urls_that_urllib3_sends_elsewhere(url: str) -> None:
    with pytest.raises(ValueError, match="not on the configured platform"):
        ensure_platform_origin(url, PLATFORM_URL)


@pytest.mark.parametrize(
    "url",
    [
        "https://api.hafnia.example@attacker.example/api/v1/experiments/1/trainer-credentials",
        "https://user:pw@api.hafnia.example/api/v1/experiments/1/trainer-credentials",
    ],
)
def test_ensure_platform_origin_rejects_userinfo(url: str) -> None:
    with pytest.raises(ValueError, match="must not contain credentials"):
        ensure_platform_origin(url, PLATFORM_URL)
