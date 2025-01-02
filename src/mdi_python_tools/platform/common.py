import os
import boto3
from botocore.exceptions import BotoCoreError, ClientError
from typing import Dict

from mdi_python_tools.dip.client import fetch
from mdi_python_tools.log import logger


def get_credentials(endpoint: str, aws_region: str) -> Dict:
    """
    Retrieve credentials for accessing the recipe stored in S3 from AWS Secrets Manager.

    Args:
        endpoint (str): The endpoint URL to fetch credentials from
        aws_region (str): AWS region where the Secrets Manager is located

    Returns:
        Dict[str, Any]: Dictionary containing the credentials with structure:
            {
                "access_key": str,
                "secret_key": str,
                "session_token": str,
                "s3_path": str
            }

    Raises:
        ValueError: If MDI_API_KEY environment variable is not set
        ClientError: If AWS Secrets Manager operation fails
        RuntimeError: If credential fetching fails
    """
    secret_name = os.getenv("MDI_API_KEY", None)
    if secret_name is None:
        raise ValueError("Environment variable 'MDI_API_KEY' is not set.")
    try:
        with boto3.client("secretsmanager", region_name=aws_region) as client:
            response = client.get_secret_value(SecretId=secret_name)
            secret_string = response["SecretString"]

            logger.debug("Successfully retrieved secret from AWS Secrets Manager")
            return fetch(endpoint, secret_string)

    except (BotoCoreError, ClientError) as e:
        logger.error(f"AWS Secrets Manager error: {e}")
        raise ClientError(
            error_response={"Error": {"Message": str(e)}},
            operation_name="get_secret_value",
        )
    except Exception as e:
        logger.error(f"Failed to fetch credentials: {e}")
        raise RuntimeError(f"Failed to retrieve credentials: {e}") from e
