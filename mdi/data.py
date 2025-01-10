import shutil

import boto3
import datasets
import requests
from tqdm import tqdm

from .config import config


class MissingDataset(Exception):
    pass


def headers_from_api_key(api_key: str) -> dict[str, str]:
    return {"X-APIKEY": api_key}


def get_dataset_obj_from_name(api_key: str, name: str) -> dict:
    """Get dataset id from name."""
    headers = headers_from_api_key(api_key)
    url = f"{config.get_api_url()}/api/v1/datasets?name__iexact={name}"
    r = requests.get(url, headers=headers)
    if r.status_code == 200:
        data = r.json()
        if len(data) == 0:
            raise MissingDataset(
                f"Dataset '{name}' appears to either not exist in " "MDI or be unavailable to you."
            )
        elif len(data) > 1:
            raise Exception(
                "This shouldn't happen found multiple datasets with same name?"
                "Report this to MDI."
            )
        dataset_id = data[0]["id"]
    else:
        r.raise_for_status()

    url = f"{config.get_api_url()}/api/v1/datasets/{dataset_id}"
    r = requests.get(url, headers=headers)
    if r.status_code == 200:
        return r.json()
    else:
        r.raise_for_status()


def get_temporary_credentials(api_key: str, obj_id: str) -> dict:
    """Fetch temporary AWS S3 credentials from the API."""
    headers = headers_from_api_key(api_key)
    url = f"{config.get_api_url()}/api/v1/datasets/{obj_id}/temporary-credentials"
    r = requests.get(url, headers=headers)
    if r.status_code == 200:
        return r.json()
    else:
        r.raise_for_status()


def load_dataset(name: str, force_redownload: bool = False) -> datasets.DatasetDict:
    """Load a dataset from AWS S3 bucket."""
    api_key = _ensure_api_key()
    dataset_obj = get_dataset_obj_from_name(api_key, name)
    dataset_obj_id = dataset_obj["id"]
    if not dataset_obj_id:
        raise ValueError(f"Unknown dataset {name}.")
    s3_bucket_name = dataset_obj["s3_bucket_name"]
    if not s3_bucket_name:
        raise ValueError(
            f"Internal error, please contact support. Missing bucket for {name} dataset."
        )

    try:
        credentials = get_temporary_credentials(api_key, dataset_obj_id)
    except requests.exceptions.HTTPError as e:
        print(e)
        raise RuntimeError("Failed to get temporary credentials.")

    load_dataset_folder = config.get_module_dir() / name
    path_local_dataset = get_or_download_dataset(
        credentials=credentials,
        bucket_name=s3_bucket_name,
        local_folder=load_dataset_folder,
        force_redownload=force_redownload,
    )
    return datasets.load_from_disk(path_local_dataset)


def get_or_download_dataset(
    credentials: dict,
    bucket_name: str,
    local_folder: str,
    force_redownload: bool,
) -> str:
    s3_client = boto3.client(
        "s3",
        aws_access_key_id=credentials["access_key"],
        aws_secret_access_key=credentials["secret_key"],
        aws_session_token=credentials["session_token"],
    )

    path_sample = local_folder / "sample"

    if path_sample.exists() and force_redownload is False:
        return str(path_sample)

    files = s3_client.list_objects_v2(Bucket=bucket_name, Prefix="sample")
    shutil.rmtree(path_sample, ignore_errors=True)
    for file in tqdm(files["Contents"]):
        file_name = file["Key"]
        path_file = local_folder / file_name
        path_file.parent.mkdir(parents=True, exist_ok=True)
        s3_client.download_file(bucket_name, file_name, path_file)

    return str(path_sample)


def list_training_runs():
    """List training runs."""
    api_key = _ensure_api_key()
    headers = headers_from_api_key(api_key)
    url = f"{config.get_api_url()}/api/v1/training-runs"
    r = requests.get(url, headers=headers)
    if r.status_code == 200:
        return r.json()
    else:
        r.raise_for_status()


def create_training_run(name: str, description: str, file):
    api_key = _ensure_api_key()
    headers = headers_from_api_key(api_key)
    url = f"{config.get_api_url()}/api/v1/training-runs"
    body = {"model_name": name, "description": description}
    r = requests.post(url, headers=headers, data=body, files=dict(recipe=file))
    if r.status_code == 201:
        return r.json()
    else:
        r.raise_for_status()


def _ensure_api_key() -> str:
    api_key = config.get_api_key()
    if not api_key:
        raise ValueError("No API key found. Please login first. Run 'mdi login' in terminal.")
    return api_key
