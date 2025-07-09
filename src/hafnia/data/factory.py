import os
from pathlib import Path
from typing import Any

from hafnia import utils
from hafnia.dataset.hafnia_dataset import HafniaDataset, get_or_create_dataset_path_from_recipe


def load_dataset(recipe: Any, force_redownload: bool = False) -> HafniaDataset:
    """Load a dataset either from a local path or from the Hafnia platform."""

    path_dataset = get_dataset_path(recipe, force_redownload=force_redownload)
    dataset = HafniaDataset.from_path(path_dataset)
    return dataset


def get_dataset_path(recipe: Any, force_redownload: bool = False) -> Path:
    if utils.is_hafnia_cloud_job():
        return Path(os.getenv("MDI_DATASET_DIR", "/opt/ml/input/data/training"))

    path_dataset = get_or_create_dataset_path_from_recipe(recipe, force_redownload=force_redownload)

    return path_dataset
