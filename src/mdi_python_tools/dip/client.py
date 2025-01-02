import json
from typing import Dict
import urllib3

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
