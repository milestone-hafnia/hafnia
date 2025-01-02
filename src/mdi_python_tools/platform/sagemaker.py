import sys
import subprocess
from typing import Dict
import boto3
from pathlib import Path

from mdi_python_tools.log import logger
from mdi_python_tools.platform.codebuild import handle_status, collect_python_modules
from mdi_python_tools.platform.common import get_credentials


def handle_launch(task: str) -> None:
    """
    Launch and execute a specified MDI task.

    This function verifies the MDI environment status, locates the task script,
    and executes it in a subprocess with output streaming.

    Args:
        task (str): Name of the task to execute

    Raises:
        ValueError: If the task is not found or scripts directory is not in PYTHONPATH
    """

    status = handle_status()
    if status.status != "ok":
        logger.warning(f"MDI environment not ready: {status}")
        return
    scripts_dir = [p for p in sys.path if "scripts" in p][0]
    scripts = collect_python_modules(Path(scripts_dir))
    if task not in scripts:
        available_tasks = ", ".join(sorted(scripts.keys()))
        raise ValueError(f"Task '{task}' not found. Available tasks: {available_tasks}")
    subprocess.check_call(
        ["python", scripts[task]["runner_path"]], stdout=sys.stdout, stderr=sys.stdout
    )


def download_s3_folder(credentials: Dict, output_dir: str) -> None:
    """
    Downloads the S3 folder specified in credentials['uri'] to the output_dir.
    """
    try:
        uri = credentials["uri"]
        if not uri.startswith("s3://"):
            raise ValueError("Invalid S3 URI format. It should start with 's3://'.")

        _, _, bucket_name, *key_parts = uri.split("/")
        key_prefix = "/".join(key_parts)

        s3_resource = boto3.resource(
            "s3",
            region_name=credentials["region"],
            aws_access_key_id=credentials["access_key"],
            aws_secret_access_key=credentials["secret_key"],
            aws_session_token=credentials["session_token"],
        )

        bucket = s3_resource.Bucket(bucket_name)
        for obj in bucket.objects.filter(Prefix=key_prefix):
            target = Path(output_dir) / Path(obj.key).relative_to(key_prefix)
            if obj.key.endswith("/"):
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            logger.info(f"Downloading {obj.key} to {target}")
            bucket.download_file(obj.key, str(target))
    except Exception as e:
        logger.error(f"Failed to download S3 folder: {e}")
        raise


def get_dataset(dataset_uuid: str, output_dir: str, aws_region: str) -> None:
    """
    Retrieves the dataset using the dataset UUID and downloads it to the specified output directory.
    """
    try:
        credentials = get_credentials(dataset_uuid, aws_region)
        download_s3_folder(credentials, output_dir)
        logger.info(f"Dataset downloaded successfully to {output_dir}")
    except Exception as e:
        logger.error(f"Failed to get dataset: {e}")
        raise
