import json
import os
from pathlib import Path
from typing import Any, Dict

import boto3
import urllib3
from botocore.exceptions import BotoCoreError, ClientError

from mdi_python_tools.config import CONFIG
from mdi_python_tools.log import logger


def fetch(endpoint: str, api_key: str) -> Dict:
    """
    Fetches data from dipdatalib backend
    Args:
        endpoint (str): The URL endpoint to fetch data from
    Returns:
        Dict: The JSON response from the endpoint
    """
    http = urllib3.PoolManager(timeout=5.0, retries=urllib3.Retry(3))
    headers = {"X-APIKEY": api_key}

    try:
        response = http.request("GET", endpoint, headers=headers)
        if response.status != 200:
            raise urllib3.exceptions.HTTPError(f"Request failed with status {response.status}")

        return json.loads(response.data.decode("utf-8"))

    except Exception as e:
        logger.error(f"Error fetching data from {endpoint}: {str(e)}")
        raise

    finally:
        http.clear()


def get_api_key() -> str:
    """
    Retrieves the MDI API key from local config or AWS Secrets Manager.

    The function will:
      1. Attempt to get the key from local configuration via CONFIG.get_api_key().
      2. If not found, it will look for the Secrets Manager name in the environment
         variable 'MDI_API_KEY'.
      3. Fetch the secret value from AWS Secrets Manager using the provided secret name
         and the region from the environment variable 'AWS_REGION' (defaulting to 'us-west-1').

    Returns:
        str: The MDI API key.

    Raises:
        ValueError: If the environment variable 'MDI_API_KEY' is not set or is empty.
        ClientError: If fetching the secret from AWS Secrets Manager fails.
    """
    local_api_key = CONFIG.get_api_key()
    if local_api_key:
        return local_api_key

    api_key_name = os.getenv("MDI_API_KEY")
    if not api_key_name:
        raise ValueError(
            "Environment variable 'MDI_API_KEY' is not set or is empty. "
            "Update it or use CLI to configure 'mdi sys configure'."
        )

    aws_region = os.getenv("AWS_REGION", "us-west-1")

    client = boto3.client("secretsmanager", region_name=aws_region)
    try:
        response = client.get_secret_value(SecretId=api_key_name)
        secret_value = response.get("SecretString")
        if not secret_value:
            # Secrets Manager can return the secret in 'SecretBinary' instead if it's not a string
            raise ValueError("Secret string was not found in the AWS Secrets Manager response.")
        return secret_value
    except (BotoCoreError, ClientError) as e:
        logger.error(f"Failed to retrieve secret from AWS Secrets Manager: {e}")
        raise ClientError(
            error_response={"Error": {"Message": str(e)}},
            operation_name="get_secret_value",
        )


def get_resource_creds(endpoint: str) -> Dict[str, Any]:
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
    api_key = get_api_key()

    try:
        logger.debug("Fetching credentials from DIP endpoint.")
        creds = fetch(endpoint, api_key)
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
    relative_path = Path(object_key)
    local_path = output_dir / relative_path
    local_path.parent.mkdir(parents=True, exist_ok=True)
    s3_client.download_file(bucket, object_key, str(local_path))
    return local_path


def download_resource(resource_url: str, destination: str) -> Dict:
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
    res_creds = get_resource_creds(resource_url)
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

            for obj in contents:
                sub_key = obj["Key"]
                local_file = download_single_object(s3_client, bucket_name, sub_key, output_path)
                downloaded_files.append(str(local_file))

            logger.info(f"Downloaded folder/prefix '{key}' with {len(downloaded_files)} object(s).")
        else:
            logger.error(f"Error checking object or prefix: {e}")
            raise RuntimeError(f"Failed to check or download S3 resource: {e}") from e

    return {"status": "success", "downloaded_files": downloaded_files}
