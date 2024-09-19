from unittest.mock import patch

import boto3
import datasets
import requests
from datasets.download.download_config import DownloadConfig

from .config import MDI_CACHE_DIR, config


def headers_from_api_key(api_key: str) -> dict[str, str]:
    return {"X-APIKEY": api_key}


def get_dataset_obj_from_name(api_key: str, name: str) -> dict:
    """Get dataset id from name."""
    headers = headers_from_api_key(api_key)
    url = f"{config.get_api_url()}/api/v1/datasets/?name__iexact={name}"
    r = requests.get(url, headers=headers)
    if r.status_code == 200:
        data = r.json()
        dataset_id = data[0]["id"]
    else:
        r.raise_for_status()

    url = f"{config.get_api_url()}/api/v1/datasets/{dataset_id}/"
    r = requests.get(url, headers=headers)
    if r.status_code == 200:
        return r.json()
    else:
        r.raise_for_status()


def get_temporary_credentials(api_key: str, obj_id: str) -> dict:
    """Fetch temporary AWS S3 credentials from the API."""
    headers = headers_from_api_key(api_key)
    url = f"{config.get_api_url()}/api/v1/datasets/{obj_id}/temporary-credentials/"
    r = requests.get(url, headers=headers)
    if r.status_code == 200:
        return r.json()
    else:
        r.raise_for_status()


def download_dataset_py_file(
    access_key: str,
    secret_key: str,
    session_token: str,
    bucket_name: str,
    local_folder: str,
) -> str:
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


def load_dataset(
    name: str, force_redownload: bool = False, verbose: bool = False
) -> datasets.Dataset:
    """Load a dataset from AWS S3 bucket."""
    api_key = _ensure_api_key()
    # get id from name
    dataset_obj = get_dataset_obj_from_name(api_key, name)
    dataset_obj_id = dataset_obj["id"]
    if not dataset_obj_id:
        raise ValueError(f"Unknown dataset {name}.")
    s3_bucket_name = dataset_obj["s3_bucket_name"]
    if not s3_bucket_name:
        raise ValueError(
            f"Internal error, please contact support. Missing bucket for {name} dataset."
        )

    # get temporary credentials
    try:
        credentials = get_temporary_credentials(api_key, dataset_obj_id)
    except requests.exceptions.HTTPError as e:
        print(e)
        raise RuntimeError("Failed to get temporary credentials.")

    # download py file
    dataset_module_folder = config.get_module_dir() / name
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

    if force_redownload:
        download_mode = datasets.DownloadMode.FORCE_REDOWNLOAD
    else:
        download_mode = datasets.DownloadMode.REUSE_DATASET_IF_EXISTS

    with patch.object(DownloadConfig, "__post_init__", lambda a, b: None):
        dataset = datasets.load_dataset(
            str(py_file_path.resolve()),
            storage_options=storage_options,
            cache_dir=MDI_CACHE_DIR,
            download_mode=download_mode,
            trust_remote_code=True,
        )

    if verbose:
        print(f"Dataset {name} downloaded successfully.")

    return dataset


def list_training_runs():
    """List training runs."""
    api_key = _ensure_api_key()
    headers = headers_from_api_key(api_key)
    url = f"{config.get_api_url()}/api/v1/training-runs/"
    r = requests.get(url, headers=headers)
    if r.status_code == 200:
        return r.json()
    else:
        r.raise_for_status()


def create_training_run(name: str, description: str, file):
    api_key = _ensure_api_key()
    headers = headers_from_api_key(api_key)
    url = f"{config.get_api_url()}/api/v1/training-runs/"
    body = {"model_name": name, "description": description}
    r = requests.post(url, headers=headers, data=body, files=dict(recipe=file))
    if r.status_code == 201:
        return r.json()
    else:
        r.raise_for_status()


def _ensure_api_key() -> str:
    api_key = config.get_api_key()
    if not api_key:
        raise ValueError(
            "No API key found. Please login first. Run 'mdi login' in terminal."
        )
    return api_key
