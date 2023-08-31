from unittest import mock

import boto3
import pytest
from botocore.stub import Stubber

from mdi import data
from mdi.config import MDI_CREDENTIALS


@pytest.fixture
def api_key():
    return "test_api_key"


@pytest.fixture
def s3_client():
    return boto3.client("s3")


def test_get_credentials(api_key):
    with mock.patch("builtins.open", mock.mock_open(read_data=api_key)) as m:
        credentials = data.get_credentials()
        m.assert_called_once_with(MDI_CREDENTIALS.resolve(), "r")
        assert credentials == api_key


def test_set_credentials(api_key):
    with mock.patch("builtins.open", mock.mock_open()) as m:
        data.set_credentials(api_key)
        m.assert_called_once_with(MDI_CREDENTIALS.resolve(), "w")


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
