from mdi_python_tools.config import CONFIG
from mdi_python_tools.http import fetch
from mdi_python_tools.utils import safe


@safe
def get_organization_id(endpoint: str) -> str:
    headers = {"X-APIKEY": CONFIG.api_key}
    org_info = fetch(endpoint, headers=headers)
    return org_info[0]["id"]
