import urllib3

from hafnia.http import fetch


def get_organization_id(endpoint: str, api_key: str) -> str:
    headers = {"X-APIKEY": api_key}
    try:
        org_info = fetch(endpoint, headers=headers)
    except urllib3.exceptions.HTTPError as e:
        raise ValueError("Failed to fetch organization ID. Verify platform URL and API key.") from e
    return org_info[0]["id"]
