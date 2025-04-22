import os
import shutil
from pathlib import Path
from typing import Optional, Union

from datasets import Dataset, DatasetDict, load_from_disk

from cli.config import Config
from hafnia import utils
from hafnia.log import logger
from hafnia.platform import download_resource, get_dataset_id


def load_local(dataset_path: Path) -> Union[Dataset, DatasetDict]:
    """Load a Hugging Face dataset from a local directory path."""
    if not dataset_path.exists():
        raise ValueError(f"Can not load dataset, directory does not exist -- {dataset_path}")
    logger.info(f"Loading data from {dataset_path.as_posix()}")
    return load_from_disk(dataset_path.as_posix())


def download_or_get_dataset_path(
    dataset_name: str,
    endpoint: str,
    api_key: str,
    output_dir: Optional[str] = None,
    force_redownload: bool = False,
) -> Path:
    """Download or get the path of the dataset."""
    output_dir = output_dir or str(utils.PATH_DATASET)
    dataset_path_base = Path(output_dir).absolute() / dataset_name
    dataset_path_base.mkdir(exist_ok=True, parents=True)
    dataset_path_sample = dataset_path_base / "sample"

    if dataset_path_sample.exists() and not force_redownload:
        logger.info("Dataset found locally. Set 'force=True' or add `--force` flag with cli to re-download")
        return dataset_path_sample

    dataset_id = get_dataset_id(dataset_name, endpoint, api_key)
    dataset_access_info_url = f"{endpoint}/{dataset_id}/temporary-credentials"

    if force_redownload and dataset_path_sample.exists():
        # Remove old files to avoid old files conflicting with new files
        shutil.rmtree(dataset_path_sample, ignore_errors=True)
    status = download_resource(dataset_access_info_url, str(dataset_path_base), api_key)
    if status:
        return dataset_path_sample
    raise RuntimeError("Failed to download dataset")


def load_from_platform(
    dataset_name: str,
    endpoint: str,
    api_key: str,
    output_dir: Optional[str] = None,
    force_redownload: bool = False,
) -> Union[Dataset, DatasetDict]:
    path_dataset = download_or_get_dataset_path(
        dataset_name=dataset_name,
        endpoint=endpoint,
        api_key=api_key,
        output_dir=output_dir,
        force_redownload=force_redownload,
    )
    return load_local(path_dataset)


def load_dataset(dataset_name: str, force_redownload: bool = False) -> Union[Dataset, DatasetDict]:
    """Load a dataset either from a local path or from the Hafnia platform."""

    if utils.is_remote_job():
        path_dataset = Path(os.getenv("MDI_DATASET_DIR", "/opt/ml/input/data/training"))
        return load_local(path_dataset)

    cfg = Config()
    endpoint_dataset = cfg.get_platform_endpoint("datasets")
    api_key = cfg.api_key
    dataset = load_from_platform(
        dataset_name=dataset_name,
        endpoint=endpoint_dataset,
        api_key=api_key,
        output_dir=None,
        force_redownload=force_redownload,
    )

    return dataset
