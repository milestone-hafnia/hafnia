from unittest import mock

import boto3
import pytest
from botocore.stub import Stubber

from mdi import data
from mdi.config import KEY_STORE_CONFIG, Config

API_URL = "https://test.com"
API_KEY = "test_api_key"
CONFIG_FILE = f"""\
[DEFAULT]
current_profile = profile.default

[profile.default]
api_url = {API_URL}
key_store = {KEY_STORE_CONFIG}
api_key = {API_KEY}
"""


@pytest.fixture
def api_key():
    return API_KEY


@pytest.fixture
def s3_client():
    return boto3.client("s3")


def test_config_read(tmp_path):
    file_path = tmp_path / "config.ini"
    # Write content to the config file.
    file_path.write_text(CONFIG_FILE)
    config = Config(file_path, tmp_path)
    assert API_URL == config.get_api_url()
    assert API_KEY == config.get_api_key()


def test_config_profile_actions(tmp_path):
    file_path = tmp_path / "config.ini"
    # Read in empty config file.
    config = Config(file_path, tmp_path)
    # Create profiles. We are using KEY_STORE_CONFIG as the key store to avoid dependency on the system's keyring.
    config.create_profile("first", "first_url", "first_key", KEY_STORE_CONFIG)
    config.create_profile("second", "second_url", "second_key", KEY_STORE_CONFIG)
    assert config.list_profiles() == {"first": "first_url", "second": "second_url"}
    assert config.get_current_profile_name() is None
    assert config.get_api_url() is None
    assert config.get_api_key() is None
    # Set one profile as current.
    config.set_current_profile("second")
    assert config.get_current_profile_name() == "second"
    assert "second_url" == config.get_api_url()
    assert "second_key" == config.get_api_key()
    # Update the profile.
    config.update_profile_api_url("second", "new_api_url")
    config.update_profile_api_key("second", "new_api_key")
    assert config.get_current_profile_name() == "second"
    assert "new_api_url" == config.get_api_url()
    assert "new_api_key" == config.get_api_key()
    # Delete the current profile.
    config.delete_profile("second")
    assert config.get_current_profile_name() is None
    assert config.get_api_url() is None
    assert config.get_api_key() is None


def test_get_temporary_credentials(api_key):
    with mock.patch("requests.get") as mocked_get:
        mocked_get.return_value.status_code = 200
        mocked_get.return_value.json.return_value = {
            "access_key": "test_access_key",
            "secret_key": "test_secret_key",
            "session_token": "test_session_token",
        }
        credentials = data.get_temporary_credentials(api_key, "test_obj_id")
        print(credentials)
        assert credentials == {
            "access_key": "test_access_key",
            "secret_key": "test_secret_key",
            "session_token": "test_session_token",
        }


def test_download_bucket_contents(s3_client):
    with Stubber(s3_client) as stubber:
        stubber.add_client_error("list_objects")
        with pytest.raises(Exception):
            data.download_bucket_contents(
                "access_key",
                "secret_key",
                "session_token",
                "bucket_name",
                "local_folder",
            )
