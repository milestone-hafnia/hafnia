from pathlib import Path
from typing import Optional, Union

from datasets import Dataset, DatasetDict, load_from_disk

from mdi_python_tools.log import logger
from mdi_python_tools.platform import download_resource, get_dataset_id


def load_local(dataset_path: Path) -> Union[Dataset, DatasetDict]:
    """Load a Hugging Face dataset from a local directory path."""
    if not dataset_path.exists():
        raise ValueError(f"Can not load dataset, directory does not exist -- {dataset_path}")
    logger.info(f"Loading data from {dataset_path.as_posix()}")
    return load_from_disk(dataset_path.as_posix())


def load_from_platform(
    dataset_name: str,
    endpoint: str,
    api_key: str,
    output_dir: Optional[str] = None,
    force: bool = False,
) -> Union[Dataset, DatasetDict]:
    """Download and load a dataset from the MDI platform, with caching for subsequent loads."""
    output_dir = "." if output_dir is None else output_dir
    dataset_path = Path(output_dir).absolute() / dataset_name
    dataset_path.parent.mkdir(exist_ok=True, parents=True)
    if dataset_path.exists() and not force:
        logger.info("Dataset found locally. Use `force=True` to re-download")
        return load_local(dataset_path / "sample")
    dataset_id = get_dataset_id(dataset_name, endpoint, api_key)
    dataset_access_info_url = f"{endpoint}/{dataset_id}/temporary-credentials"
    status = download_resource(dataset_access_info_url, dataset_path, api_key)
    if status:
        return load_local(dataset_path / "sample")
    raise NotImplementedError("Download dataset is not implemented yet.")


def load_dataset(
    data_path: Optional[str] = None,
    mdi_platform_endpoint: Optional[str] = None,
    mdi_dataset_name: Optional[str] = None,
    output_dir: Optional[str] = None,
    api_key: Optional[str] = None,
) -> Union[Dataset, DatasetDict]:
    """Load a dataset either from a local path or from the MDI platform."""
    if data_path is not None:
        return load_local(Path(data_path))
    if mdi_dataset_name is not None:
        if api_key is None or mdi_platform_endpoint is None:
            raise ValueError("Provide `api_key` in order to use `mdi_dataset_name`.")
        return load_from_platform(mdi_dataset_name, mdi_platform_endpoint, api_key, output_dir)
    raise ValueError("Please provide dataset_path or mdi_dataset_name")
