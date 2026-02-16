from pathlib import Path

import urllib3

from hafnia import http
from hafnia_cli.config import Config

if __name__ == "__main__":
    cfg = Config()

    endpoint = cfg.get_platform_endpoint("assets")
    path_assets = Path(".data/assets")
    path_assets.mkdir(parents=True, exist_ok=True)
    path_asset = path_assets / "my_first_asset.mp4"
    path_asset.write_text("This is my first asset!")

    headers = {"Authorization": cfg.api_key, "accept": "application/json"}
    data = {"file": (path_asset.name, path_asset.read_bytes()), "name": "My First Asset"}
    response = http.post(endpoint, headers=headers, data=data, multipart=True)

    print(f"Upload successful! Response: {response}")

    # Alternative: Direct urllib3 implementation without http module
    print("\n--- Alternative implementation using urllib3 directly ---")

    http_client = urllib3.PoolManager(retries=urllib3.Retry(3))
    try:
        # For multipart, urllib3 automatically sets Content-Type header
        response2 = http_client.request(
            "POST",
            endpoint,
            fields={"file": (path_asset.name, path_asset.read_bytes()), "name": "My Second Asset (Direct Upload)"},
            headers={"Authorization": cfg.api_key, "accept": "application/json"},
        )

        if response2.status in (200, 201):
            import json

            result = json.loads(response2.data.decode("utf-8"))
            print(f"Direct upload successful! Response: {result}")
        else:
            print(f"Upload failed with status {response2.status}: {response2.data.decode('utf-8')}")
    finally:
        http_client.clear()

    # headers = {"Authorization": api_key, "accept": "application/json"}
    # data = {
    #     "name": path_trainer.name,
    #     "description": "Trainer package created by Hafnia CLI",
    #     "file": (zip_path.name, Path(zip_path).read_bytes()),
    # }
    # response = http.post(endpoint, headers=headers, data=data, multipart=True)
