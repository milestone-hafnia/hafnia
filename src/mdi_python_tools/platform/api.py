from typing import Optional

from mdi_python_tools.config import CONFIG
from mdi_python_tools.http import fetch
from mdi_python_tools.log import logger


def get_organization_id(endpoint: str) -> Optional[str]:
    headers = {"X-APIKEY": CONFIG.api_key}
    try:
        org_info = fetch(endpoint, headers=headers)
    except Exception as e:
        logger.error(f"Error fetching organization info: {str(e)}")
        return None
    return org_info[0]["id"]
