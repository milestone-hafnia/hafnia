import os
from unittest.mock import patch

import boto3
import datasets
import requests
from datasets.download.download_config import DownloadConfig

from .config import (
    MDI_API_DATASETS,
    MDI_API_TEMP_CREDS,
    MDI_CACHE_DIR,
    MDI_CREDENTIALS,
    MDI_MODULE_DIR,
)


def get_credentials():
    """Get the API key from the credentials file."""
    if MDI_CREDENTIALS.exists():
        with open(MDI_CREDENTIALS, "r") as f:
            api_key = f.read().strip()
        return api_key
    else:
        return None


def set_credentials(api_key):
    """Set the API key in the credentials file."""
    credentials_folder = MDI_CREDENTIALS.parent

    # Make sure the .mdi folder exists
    if not credentials_folder.exists():
        credentials_folder.mkdir(parents=True, exist_ok=True)

    with open(MDI_CREDENTIALS, "w") as f:
        f.write(api_key)

    # Set the file permissions to 600
    os.chmod(MDI_CREDENTIALS, 0o600)


def headers_from_api_key(api_key):
    return {"X-APIKEY": api_key}


def get_dataset_obj_from_name(api_key, name):
    """Get dataset id from name."""
    headers = headers_from_api_key(api_key)
    url = MDI_API_DATASETS.format(name=name)
    r = requests.get(url, headers=headers)
    if r.status_code == 200:
        data = r.json()
        return data[0]
    else:
        r.raise_for_status()


def get_temporary_credentials(api_key, obj_id):
    """Fetch temporary AWS S3 credentials from the API."""
    headers = headers_from_api_key(api_key)
    url = MDI_API_TEMP_CREDS.format(obj_id=obj_id)
    r = requests.get(url, headers=headers)
    if r.status_code == 200:
        return r.json()
    else:
        r.raise_for_status()


def download_dataset_py_file(
    access_key, secret_key, session_token, bucket_name, local_folder
):
    # Create a session using the temporary credentials
    session = boto3.Session(
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        aws_session_token=session_token,
    )

    # Create a S3 resource from the session
    s3 = session.resource("s3")

    # Download each file from the bucket
    bucket = s3.Bucket(bucket_name)

    # Find py file in bucket root
    py_file = None
    for obj in bucket.objects.all():
        if obj.key.endswith(".py"):
            py_file = obj.key
            break

    # Download py file
    if py_file:
        py_file_path = local_folder / py_file
        bucket.download_file(py_file, py_file_path)
        return py_file_path
    else:
        raise RuntimeError("No py file found in bucket root.")


def load_dataset(name, force_download=False, verbose=False):
    """Load a dataset from AWS S3 bucket."""
    api_key = get_credentials()
    if not api_key:
        raise ValueError("No API key found. Please login first.")

    # get id from name
    dataset_obj = get_dataset_obj_from_name(api_key, name)
    dataset_obj_id = dataset_obj["id"]
    s3_bucket_name = dataset_obj["s3_bucket_name"]

    # get temporary credentials
    try:
        credentials = get_temporary_credentials(api_key, dataset_obj_id)
    except requests.exceptions.HTTPError as e:
        print(e)
        raise RuntimeError("Failed to get temporary credentials.")

    # download py file
    dataset_module_folder = MDI_MODULE_DIR / name
    dataset_module_folder.mkdir(parents=True, exist_ok=True)
    py_file_path = download_dataset_py_file(
        credentials["access_key"],
        credentials["secret_key"],
        credentials["session_token"],
        s3_bucket_name,
        dataset_module_folder,
    )

    storage_options = {
        "key": credentials["access_key"],
        "secret": credentials["secret_key"],
        "token": credentials["session_token"],
    }

    # use unittest patch to mock the __post_init__ method
    # to make sure the "hf" key is never written
    with patch.object(DownloadConfig, "__post_init__", lambda a, b: None):
        download_config = DownloadConfig(
            storage_options=storage_options,
            cache_dir=MDI_CACHE_DIR,
            force_download=force_download,
        )

        dataset = datasets.load_dataset(
            str(py_file_path.resolve()),
            download_config=download_config,
        )

    if verbose:
        print(f"Dataset {name} downloaded successfully.")

    return dataset
