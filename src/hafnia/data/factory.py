import os
from pathlib import Path
from typing import Any

from hafnia import utils
from hafnia.dataset.hafnia_dataset import HafniaDataset


def load_dataset(data_recipe: Any, force_redownload: bool = False) -> HafniaDataset:
    """Load a dataset either from a local path or from the Hafnia platform."""

    path_dataset = get_dataset_path(data_recipe, force_redownload=force_redownload)
    dataset = HafniaDataset.from_path(path_dataset)
    return dataset


def get_dataset_path(data_recipe: Any, force_redownload: bool = False) -> Path:
    if utils.is_remote_job():
        return Path(os.getenv("MDI_DATASET_DIR", "/opt/ml/input/data/training"))

    path_dataset = HafniaDataset.from_recipe_to_disk(data_recipe, force_redownload=force_redownload)

    return path_dataset
