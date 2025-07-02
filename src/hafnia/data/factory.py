import os
import shutil
from pathlib import Path
from typing import Any

from hafnia import utils
from hafnia.dataset.hafnia_dataset import HafniaDataset


def load_dataset(dataset_name: Any, force_redownload: bool = False) -> HafniaDataset:
    """Load a dataset either from a local path or from the Hafnia platform."""

    path_dataset = get_dataset_path(dataset_name, force_redownload=force_redownload)
    dataset = HafniaDataset.from_path(path_dataset)
    return dataset


def get_dataset_path(dataset_name: Any, force_redownload: bool = False) -> Path:
    from hafnia.dataset.builder.builder_helpers import (
        convert_to_explicit_specification,
        get_dataset_path_from_specification,
    )
    from hafnia.dataset.builder.builders import DatasetBuilder

    if utils.is_remote_job():
        return Path(os.getenv("MDI_DATASET_DIR", "/opt/ml/input/data/training"))

    specification_explicit: DatasetBuilder = convert_to_explicit_specification(dataset_name)
    path_dataset = get_dataset_path_from_specification(specification_explicit)

    if force_redownload:
        shutil.rmtree(path_dataset, ignore_errors=True)

    if HafniaDataset.check_dataset_path(path_dataset, raise_error=False):
        return path_dataset
    path_dataset.mkdir(parents=True, exist_ok=True)

    path_specification_json = path_dataset / "specification.json"
    path_specification_json.write_text(specification_explicit.model_dump_json(indent=4))

    dataset: HafniaDataset = specification_explicit()  # Build dataset
    dataset.write(path_dataset)
    return path_dataset
