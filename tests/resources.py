from pathlib import Path

from hafnia.platform.datasets import get_dataset_id
from hafnia.platform.download import download_resource
from hafnia_cli.config import Config


def test_download_resource(tmp_path: Path):
    cfg = Config()
    dataset_name = "tiny-dataset"

    endpoint_dataset = cfg.get_platform_endpoint("datasets")
    dataset_id = get_dataset_id(dataset_name=dataset_name, cfg=cfg)
    credentials_endpoint_suffix = "temporary-credentials"  # Access to sample dataset
    access_dataset_endpoint = f"{endpoint_dataset}/{dataset_id}/{credentials_endpoint_suffix}"

    status = download_resource(
        resource_url=access_dataset_endpoint,
        destination=str(tmp_path),
        api_key=cfg.api_key,
        prefix="sample",  # Download only a subset of files
    )
    status["status"] == "success"


def test_asdf_download_resource(tmp_path: Path):
    cfg = Config()
    dataset_name = "tiny-dataset"
    # api/v1/dataset-recipe-templates/901c9a1c-6fab-484a-83c0-2e68ffc2efcc%22
    endpoint_dataset = cfg.get_platform_endpoint("trainer")
    dataset_id = get_dataset_id(dataset_name=dataset_name, cfg=cfg)
    credentials_endpoint_suffix = "temporary-credentials"  # Access to sample dataset
    access_dataset_endpoint = f"{endpoint_dataset}/{dataset_id}/{credentials_endpoint_suffix}"

    status = download_resource(
        resource_url=access_dataset_endpoint,
        destination=str(tmp_path),
        api_key=cfg.api_key,
        prefix="sample",  # Download only a subset of files
    )
    status["status"] == "success"
