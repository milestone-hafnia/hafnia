from pathlib import Path
from typing import Dict

import boto3
from botocore.exceptions import ClientError
from pydantic import BaseModel, field_validator
from tqdm import tqdm

from hafnia.http import fetch
from hafnia.log import sys_logger, user_logger

ARN_PREFIX = "arn:aws:s3:::"


class ResourceCredentials(BaseModel):
    access_key: str
    secret_key: str
    session_token: str
    s3_arn: str
    region: str

    @staticmethod
    def fix_naming(payload: Dict[str, str]) -> "ResourceCredentials":
        """
        The endpoint returns a payload with a key called 's3_path', but it
        is actually an ARN path (starts with arn:aws:s3::). This method renames it to 's3_arn' for consistency.
        """
        if "s3_path" in payload and payload["s3_path"].startswith(ARN_PREFIX):
            payload["s3_arn"] = payload.pop("s3_path")

        if "region" not in payload:
            payload["region"] = "eu-west-1"
        return ResourceCredentials(**payload)

    @field_validator("s3_arn")
    @classmethod
    def validate_s3_arn(cls, value: str) -> str:
        """Validate s3_arn to ensure it starts with 'arn:aws:s3:::'"""
        if not value.startswith("arn:aws:s3:::"):
            raise ValueError(f"Invalid S3 ARN: {value}. It should start with 'arn:aws:s3:::'")
        return value

    def s3_path(self) -> str:
        """
        Extracts the S3 path from the ARN.
        Example: arn:aws:s3:::my-bucket/my-prefix -> my-bucket/my-prefix
        """
        return self.s3_arn[len(ARN_PREFIX) :]

    def s3_uri(self) -> str:
        """
        Converts the S3 ARN to a URI format.
        Example: arn:aws:s3:::my-bucket/my-prefix -> s3://my-bucket/my-prefix
        """
        return f"s3://{self.s3_path()}"

    def bucket_name(self) -> str:
        """
        Extracts the bucket name from the S3 ARN.
        Example: arn:aws:s3:::my-bucket/my-prefix -> my-bucket
        """
        return self.s3_path().split("/")[0]

    def object_key(self) -> str:
        """
        Extracts the object key from the S3 ARN.
        Example: arn:aws:s3:::my-bucket/my-prefix -> my-prefix
        """
        return "/".join(self.s3_path().split("/")[1:])

    def aws_credentials(self) -> Dict[str, str]:
        """
        Returns the AWS credentials as a dictionary.
        """
        environment_vars = {
            "AWS_ACCESS_KEY_ID": self.access_key,
            "AWS_SECRET_ACCESS_KEY": self.secret_key,
            "AWS_SESSION_TOKEN": self.session_token,
            "AWS_REGION": self.region,
        }
        return environment_vars


def get_resource_credentials(endpoint: str, api_key: str) -> ResourceCredentials:
    """
    Retrieve credentials for accessing the recipe stored in S3 (or another resource)
    by calling a DIP endpoint with the API key.

    Args:
        endpoint (str): The endpoint URL to fetch credentials from.

    Returns:
        ResourceCredentials

    Raises:
        RuntimeError: If the call to fetch the credentials fails for any reason.
    """
    try:
        credentials_dict = fetch(endpoint, headers={"Authorization": api_key, "accept": "application/json"})
        credentials = ResourceCredentials.fix_naming(credentials_dict)
        sys_logger.debug("Successfully retrieved credentials from DIP endpoint.")
        return credentials
    except Exception as e:
        sys_logger.error(f"Failed to fetch credentials from endpoint: {e}")
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
    res_credentials = get_resource_credentials(resource_url, api_key)

    bucket_name = res_credentials.bucket_name()
    key = res_credentials.object_key()

    output_path = Path(destination)
    output_path.mkdir(parents=True, exist_ok=True)
    s3_client = boto3.client(
        "s3",
        aws_access_key_id=res_credentials.access_key,
        aws_secret_access_key=res_credentials.secret_key,
        aws_session_token=res_credentials.session_token,
    )
    downloaded_files = []
    try:
        s3_client.head_object(Bucket=bucket_name, Key=key)
        local_file = download_single_object(s3_client, bucket_name, key, output_path)
        downloaded_files.append(str(local_file))
        user_logger.info(f"Downloaded single file: {local_file}")

    except ClientError as e:
        error_code = e.response.get("Error", {}).get("Code")
        if error_code == "404":
            sys_logger.debug(f"Object '{key}' not found; trying as a prefix.")
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

            user_logger.info(f"Downloaded folder/prefix '{key}' with {len(downloaded_files)} object(s).")
        else:
            user_logger.error(f"Error checking object or prefix: {e}")
            raise RuntimeError(f"Failed to check or download S3 resource: {e}") from e

    return {"status": "success", "downloaded_files": downloaded_files}
