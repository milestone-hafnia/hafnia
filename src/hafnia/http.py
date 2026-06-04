import json
from pathlib import Path
from typing import Dict, Optional, Sequence, Union

import urllib3

# Default per-request timeout (seconds). urllib3 has no timeout by default, which
# means a hung server would block the CLI indefinitely.
DEFAULT_TIMEOUT = 30.0

# Retry only transient failures, and only in a way that's safe to repeat:
#   - backoff_factor gives the transient condition (network blip, LB cold start)
#     time to clear between attempts — without it, retries fire instantly and
#     tend to hit the same failure.
#   - status_forcelist covers transient server/LB errors; deterministic 4xx are
#     not retried (they'd fail identically).
#   - allowed_methods is left at urllib3's default, which excludes POST/PATCH so
#     non-idempotent requests are never replayed (no double-submit).
_RETRIES = urllib3.Retry(
    total=3,
    backoff_factor=0.5,  # waits ~0.5s, 1s, 2s between attempts
    status_forcelist=(502, 503, 504),
)

# A single shared pool manager so connections (and TLS handshakes) are reused
# across requests instead of being rebuilt on every call. urllib3's PoolManager
# is thread-safe.
_http = urllib3.PoolManager(retries=_RETRIES, timeout=DEFAULT_TIMEOUT)

# Generic, endpoint-independent messages for statuses that mean the same thing
# everywhere. Callers get these for free; they can override or extend per call
# via the `status_messages` argument. Domain-specific wording (e.g. which
# resource was missing on a 404) belongs at the call site, not here.
_DEFAULT_MESSAGES: Dict[int, str] = {
    401: "Authentication failed — check your API key (try `hafnia configure`).",
    403: "You don't have permission to access this resource.",
    503: "The Hafnia platform is temporarily unavailable — please try again shortly.",
}


class HafniaHTTPError(urllib3.exceptions.HTTPError):
    """Raised when a request returns a status outside its accepted set.

    Subclasses urllib3's ``HTTPError`` so existing ``except urllib3.exceptions.HTTPError``
    handlers keep working. Carries the structured ``status_code`` (plus ``body``,
    ``method`` and ``url``) so callers can branch — e.g. treat 404 as "not found"
    and return ``None`` — instead of parsing the message string.
    """

    def __init__(self, message: str, *, status_code: int, body: str, method: str, url: str):
        self.status_code = status_code
        self.body = body
        self.method = method
        self.url = url
        super().__init__(message)


def _prepare_body(data: Union[Path, Dict, bytes, str], headers: Dict) -> Union[bytes, str]:
    """Convert a non-multipart payload into a request body, setting Content-Type as needed.

    Mutates the (already-copied) ``headers`` dict for JSON payloads.
    """
    if isinstance(data, Path):
        return data.read_bytes()
    if isinstance(data, (str, dict)):
        headers["Content-Type"] = "application/json"
        return data if isinstance(data, str) else json.dumps(data)
    if isinstance(data, bytes):
        return data
    raise ValueError(f"Unsupported data type: {type(data)}")


def _request(
    method: str,
    endpoint: str,
    headers: Dict,
    *,
    ok_statuses: Sequence[int],
    status_messages: Optional[Dict[int, str]] = None,
    body: Optional[Union[bytes, str]] = None,
    fields: Optional[Dict] = None,
    allow_empty: bool = False,
) -> Dict:
    """Issue an HTTP request and decode the JSON response.

    Args:
        method: HTTP verb (GET/POST/PATCH/DELETE).
        endpoint: Full request URL.
        headers: Request headers.
        ok_statuses: Status codes treated as success; anything else raises.
        status_messages: Optional caller-supplied ``{status: message}`` overrides, layered
            on top of ``_DEFAULT_MESSAGES``. Lets a caller attach context-specific wording
            (e.g. ``{404: "Dataset 'kitti' not found."}``) without try/except boilerplate.
        body: Raw request body (mutually exclusive with ``fields``).
        fields: Query params (GET) or multipart fields (mutually exclusive with ``body``).
        allow_empty: When True, an empty response body returns ``{}`` instead of failing to parse.

    Returns:
        The decoded JSON response, or ``{}`` for an empty body when ``allow_empty`` is set.

    Raises:
        HafniaHTTPError: On a status code outside ``ok_statuses``.
        json.JSONDecodeError: On a non-empty, non-JSON response body.
    """
    response = _http.request(method, endpoint, body=body, fields=fields, headers=headers)
    if response.status not in ok_statuses:
        error_body = response.data.decode("utf-8")
        messages = {**_DEFAULT_MESSAGES, **(status_messages or {})}
        message = messages.get(response.status, f"Request failed with status {response.status}: {error_body}")
        raise HafniaHTTPError(message, status_code=response.status, body=error_body, method=method, url=endpoint)
    if allow_empty and not response.data:
        return {}
    return json.loads(response.data.decode("utf-8"))


def fetch(
    endpoint: str,
    headers: Dict,
    params: Optional[Dict] = None,
    *,
    ok_statuses: Sequence[int] = (200,),
    status_messages: Optional[Dict[int, str]] = None,
) -> Dict:
    """Fetches data from the API endpoint.

    Args:
        endpoint: API endpoint URL
        headers: Request headers
        params: Optional query parameters
        ok_statuses: Status codes treated as success (default: 200)
        status_messages: Optional ``{status: message}`` overrides for context-specific errors

    Returns:
        Dict: JSON response from the API

    Raises:
        HafniaHTTPError: On a status code outside ``ok_statuses``
        json.JSONDecodeError: On invalid JSON response
    """
    return _request(
        "GET", endpoint, headers, ok_statuses=ok_statuses, status_messages=status_messages, fields=params or {}
    )


def post(
    endpoint: str,
    headers: Dict,
    data: Union[Path, Dict, bytes, str],
    multipart: bool = False,
    *,
    ok_statuses: Sequence[int] = (200, 201),
    status_messages: Optional[Dict[int, str]] = None,
) -> Dict:
    """Posts data to backend endpoint.

    Args:
        endpoint: The URL endpoint to post data to
        headers: Request headers
        data: The data to post - either a file path, dictionary, string or raw bytes
        multipart: Whether to send as multipart/form-data
        ok_statuses: Status codes treated as success (default: 200, 201)
        status_messages: Optional ``{status: message}`` overrides for context-specific errors

    Returns:
        Dict: The JSON response from the endpoint

    Raises:
        FileNotFoundError: If the provided Path doesn't exist
        HafniaHTTPError: If the request fails
        json.JSONDecodeError: If response isn't valid JSON
        ValueError: If data type is unsupported
    """
    headers = dict(headers)  # copy so we never mutate the caller's dict
    if multipart:
        # urllib3 sets the multipart Content-Type (with boundary) itself.
        headers.pop("Content-Type", None)
        fields = data if isinstance(data, dict) else {"file": data}
        return _request(
            "POST", endpoint, headers, ok_statuses=ok_statuses, status_messages=status_messages, fields=fields
        )
    body = _prepare_body(data, headers)
    return _request("POST", endpoint, headers, ok_statuses=ok_statuses, status_messages=status_messages, body=body)


def patch(
    endpoint: str,
    headers: Dict,
    data: Union[Path, Dict, bytes, str],
    multipart: bool = False,
    *,
    ok_statuses: Sequence[int] = (200, 202),
    status_messages: Optional[Dict[int, str]] = None,
) -> Dict:
    """Sends a PATCH request to the specified endpoint.

    Mirrors `post`: supports JSON dict/str/bytes payloads or multipart/form-data.

    Args:
        ok_statuses: Status codes treated as success (default: 200, 202)
        status_messages: Optional ``{status: message}`` overrides for context-specific errors

    Raises:
        HafniaHTTPError: If the request fails
        json.JSONDecodeError: If response isn't valid JSON
        ValueError: If data type is unsupported
    """
    headers = dict(headers)
    if multipart:
        headers.pop("Content-Type", None)
        fields = data if isinstance(data, dict) else {"file": data}
        return _request(
            "PATCH", endpoint, headers, ok_statuses=ok_statuses, status_messages=status_messages, fields=fields
        )
    body = _prepare_body(data, headers)
    return _request("PATCH", endpoint, headers, ok_statuses=ok_statuses, status_messages=status_messages, body=body)


def delete(
    endpoint: str,
    headers: Dict,
    *,
    ok_statuses: Sequence[int] = (200, 204),
    status_messages: Optional[Dict[int, str]] = None,
) -> Dict:
    """Sends a DELETE request to the specified endpoint.

    Args:
        ok_statuses: Status codes treated as success (default: 200, 204)
        status_messages: Optional ``{status: message}`` overrides for context-specific errors

    Raises:
        HafniaHTTPError: If the request fails
    """
    return _request(
        "DELETE", endpoint, headers, ok_statuses=ok_statuses, status_messages=status_messages, allow_empty=True
    )
