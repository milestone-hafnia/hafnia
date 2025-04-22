from pathlib import Path
from typing import Any, Dict

import boto3
from botocore.exceptions import ClientError
from tqdm import tqdm

from hafnia.http import fetch
from hafnia.log import logger


def get_resource_creds(endpoint: str, api_key: str) -> Dict[str, Any]:
    """
    Retrieve credentials for accessing the recipe stored in S3 (or another resource)
    by calling a DIP endpoint with the API key.

    Args:
        endpoint (str): The endpoint URL to fetch credentials from.

    Returns:
        Dict[str, Any]: Dictionary containing the credentials, for example:
            {
                "access_key": str,
                "secret_key": str,
                "session_token": str,
                "s3_path": str
            }

    Raises:
        RuntimeError: If the call to fetch the credentials fails for any reason.
    """
    try:
        creds = fetch(endpoint, headers={"X-APIKEY": api_key, "accept": "application/json"})
        logger.debug("Successfully retrieved credentials from DIP endpoint.")
        return creds
    except Exception as e:
        logger.error(f"Failed to fetch credentials from endpoint: {e}")
        raise RuntimeError(f"Failed to retrieve credentials: {e}") from e


def download_single_object(s3_client, bucket: str, object_key: str, output_dir: Path) -> Path:
    """
    Downloads a single object from S3 to a local path.

    Args:
        s3_client: The Boto3 S3 client.
        bucket (str): S3 bucket name.
        object_key (str): The S3 object key to download.
        output_dir (Path): The local directory in which to place the file.

    Returns:
        Path: The local path where the file was saved.
    """
    local_path = output_dir / object_key
    local_path.parent.mkdir(parents=True, exist_ok=True)
    s3_client.download_file(bucket, object_key, local_path.as_posix())
    return local_path


def download_resource(resource_url: str, destination: str, api_key: str) -> Dict:
    """
    Downloads either a single file from S3 or all objects under a prefix.

    Args:
        resource_url (str): The URL or identifier used to fetch S3 credentials.
        destination (str): Path to local directory where files will be stored.

    Returns:
        Dict[str, Any]: A dictionary containing download info, e.g.:
            {
                "status": "success",
                "downloaded_files": ["/path/to/file", "/path/to/other"]
            }

    Raises:
        ValueError: If the S3 ARN is invalid or no objects found under prefix.
        RuntimeError: If S3 calls fail with an unexpected error.
    """
    res_creds = get_resource_creds(resource_url, api_key)
    s3_arn = res_creds["s3_path"]
    arn_prefix = "arn:aws:s3:::"
    if not s3_arn.startswith(arn_prefix):
        raise ValueError(f"Invalid S3 ARN: {s3_arn}")

    s3_path = s3_arn[len(arn_prefix) :]
    bucket_name, *key_parts = s3_path.split("/")
    key = "/".join(key_parts)

    output_path = Path(destination)
    output_path.mkdir(parents=True, exist_ok=True)
    s3_client = boto3.client(
        "s3",
        aws_access_key_id=res_creds["access_key"],
        aws_secret_access_key=res_creds["secret_key"],
        aws_session_token=res_creds["session_token"],
    )
    downloaded_files = []
    try:
        s3_client.head_object(Bucket=bucket_name, Key=key)
        local_file = download_single_object(s3_client, bucket_name, key, output_path)
        downloaded_files.append(str(local_file))
        logger.info(f"Downloaded single file: {local_file}")

    except ClientError as e:
        error_code = e.response.get("Error", {}).get("Code")
        if error_code == "404":
            logger.debug(f"Object '{key}' not found; trying as a prefix.")
            response = s3_client.list_objects_v2(Bucket=bucket_name, Prefix=key)
            contents = response.get("Contents", [])

            if not contents:
                raise ValueError(f"No objects found for prefix '{key}' in bucket '{bucket_name}'")
            pbar = tqdm(contents)
            for obj in pbar:
                sub_key = obj["Key"]
                size_mb = obj.get("Size", 0) / 1024 / 1024
                pbar.set_description(f"{sub_key} ({size_mb:.2f} MB)")
                local_file = download_single_object(s3_client, bucket_name, sub_key, output_path)
                downloaded_files.append(local_file.as_posix())

            logger.info(f"Downloaded folder/prefix '{key}' with {len(downloaded_files)} object(s).")
        else:
            logger.error(f"Error checking object or prefix: {e}")
            raise RuntimeError(f"Failed to check or download S3 resource: {e}") from e

    return {"status": "success", "downloaded_files": downloaded_files}
