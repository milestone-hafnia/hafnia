import os
import shutil
from pathlib import Path
from typing import Optional

from cli.config import Config
from hafnia import utils
from hafnia.dataset.hafnia_dataset import HafniaDataset
from hafnia.log import user_logger
from hafnia.platform import download_resource, get_dataset_id


def download_or_get_dataset_path(
    dataset_name: str,
    cfg: Optional[Config] = None,
    output_dir: Optional[str] = None,
    force_redownload: bool = False,
) -> Path:
    """Download or get the path of the dataset."""

    cfg = cfg or Config()
    endpoint_dataset = cfg.get_platform_endpoint("datasets")
    api_key = cfg.api_key

    output_dir = output_dir or str(utils.PATH_DATASETS)
    dataset_path_base = Path(output_dir).absolute() / dataset_name
    dataset_path_base.mkdir(exist_ok=True, parents=True)
    dataset_path_sample = dataset_path_base / "sample"

    if dataset_path_sample.exists() and not force_redownload:
        user_logger.info("Dataset found locally. Set 'force=True' or add `--force` flag with cli to re-download")
        return dataset_path_sample

    dataset_id = get_dataset_id(dataset_name, endpoint_dataset, api_key)
    dataset_access_info_url = f"{endpoint_dataset}/{dataset_id}/temporary-credentials"

    if force_redownload and dataset_path_sample.exists():
        # Remove old files to avoid old files conflicting with new files
        shutil.rmtree(dataset_path_sample, ignore_errors=True)
    status = download_resource(dataset_access_info_url, str(dataset_path_base), api_key)
    if status:
        return dataset_path_sample
    raise RuntimeError("Failed to download dataset")


def get_dataset_path(dataset_name: str, force_redownload: bool = False) -> Path:
    if utils.is_remote_job():
        return Path(os.getenv("MDI_DATASET_DIR", "/opt/ml/input/data/training"))

    path_dataset = download_or_get_dataset_path(
        dataset_name=dataset_name,
        force_redownload=force_redownload,
    )
    return path_dataset


def load_dataset(dataset_name: str, force_redownload: bool = False) -> HafniaDataset:
    """Load a dataset either from a local path or from the Hafnia platform."""

    path_dataset = get_dataset_path(dataset_name, force_redownload=force_redownload)
    dataset = HafniaDataset.read_from_path(path_dataset)
    return dataset
