from pathlib import Path

from hafnia.dataset.hafnia_dataset import HafniaDataset
from hafnia.platform.datasets import download_or_get_dataset_path


def load_dataset(dataset_name: str, force_redownload: bool = False) -> HafniaDataset:
    """Load a dataset either from a local path or from the Hafnia platform."""

    path_dataset = get_dataset_path(dataset_name, force_redownload=force_redownload)
    dataset = HafniaDataset.read_from_path(path_dataset)
    return dataset


def get_dataset_path(dataset_name: str, force_redownload: bool = False) -> Path:
    path_dataset = download_or_get_dataset_path(
        dataset_name=dataset_name,
        force_redownload=force_redownload,
    )
    return path_dataset
