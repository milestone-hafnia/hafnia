import hashlib
import shutil
from pathlib import Path
from random import Random
from typing import TYPE_CHECKING, Callable, Dict

import cv2
import numpy as np
import polars as pl
from PIL import Image
from tqdm import tqdm

from hafnia.dataset import dataset_helpers

if TYPE_CHECKING:
    from hafnia.dataset.hafnia_dataset import HafniaDataset


### Image transformations ###
class AnonymizeByPixelation:
    def __init__(self, resize_factor: float = 0.10):
        self.resize_factor = resize_factor

    def __call__(self, frame: np.ndarray) -> np.ndarray:
        org_size = frame.shape[:2]
        frame = cv2.resize(frame, (0, 0), fx=self.resize_factor, fy=self.resize_factor)
        frame = cv2.resize(frame, org_size[::-1], interpolation=cv2.INTER_NEAREST)
        return frame


def splits_by_ratios(dataset: "HafniaDataset", split_ratios: Dict[str, float], seed: int = 42) -> "HafniaDataset":
    n_items = len(dataset)
    split_name_column = dataset_helpers.split_names_from_ratios(split_ratios=split_ratios, n_items=n_items, seed=seed)
    table = dataset.table.with_columns(pl.Series(split_name_column).alias("split"))
    return dataset.update_table(table)


def shuffle_dataset(dataset: "HafniaDataset", seed: int = 42) -> "HafniaDataset":
    table = dataset.table.sample(n=len(dataset), with_replacement=False, seed=seed, shuffle=True)
    return dataset.update_table(table)


def sample(dataset: "HafniaDataset", n_samples: int, shuffle: bool = True, seed: int = 42) -> "HafniaDataset":
    table = dataset.table.sample(n=n_samples, with_replacement=False, seed=seed, shuffle=shuffle)
    return dataset.update_table(table)


def define_sample_set_by_size(dataset: "HafniaDataset", n_samples: int, seed: int = 42) -> "HafniaDataset":
    is_sample_indices = Random(seed).sample(range(len(dataset)), n_samples)
    is_sample_column = [False for _ in range(len(dataset))]
    for idx in is_sample_indices:
        is_sample_column[idx] = True

    table = dataset.table.with_columns(pl.Series(is_sample_column).alias("is_sample"))
    return dataset.update_table(table)


def transform_images(
    dataset: "HafniaDataset", transform: Callable[[np.ndarray], np.ndarray], path_output: Path
) -> "HafniaDataset":
    new_paths = []
    for org_path in tqdm(dataset.table["file_name"].to_list(), desc="Transform images"):
        org_path = Path(org_path)
        if not org_path.exists():
            raise FileNotFoundError(f"File {org_path} does not exist in the dataset.")

        new_path = path_output / "data" / org_path.name
        if not new_path.parent.exists():
            new_path.parent.mkdir(parents=True, exist_ok=True)
        image = np.array(Image.open(org_path))
        image_transformed = transform(image)
        Image.fromarray(image_transformed).save(new_path)

        if not new_path.exists():
            raise FileNotFoundError(f"Transformed file {new_path} does not exist in the dataset.")
        new_paths.append(str(new_path))

    table = dataset.table.with_columns(pl.Series(new_paths).alias("file_name"))
    return dataset.update_table(table)


def rename_to_unique_image_names(dataset: "HafniaDataset", path_output: Path) -> "HafniaDataset":
    print(f"Copy images to have unique filenames. New path is '{path_output}'")
    shutil.rmtree(path_output, ignore_errors=True)  # Remove the output folder if it exists
    new_paths = []
    for org_path in tqdm(dataset.table["file_name"].to_list(), desc="- Rename/copy images"):
        org_path = Path(org_path)
        if not org_path.exists():
            raise FileNotFoundError(f"File {org_path} does not exist in the dataset.")

        hash_name = hashlib.md5(str(org_path).encode()).hexdigest()[
            :6
        ]  # Generate a unique name based on the original file name
        new_path = path_output / "data" / f"{hash_name}_{org_path.name}"
        if not new_path.parent.exists():
            new_path.parent.mkdir(parents=True, exist_ok=True)

        shutil.copyfile(org_path, new_path)  # Copy the original file to the new path
        new_paths.append(str(new_path))

    table = dataset.table.with_columns(pl.Series(new_paths).alias("file_name"))
    return dataset.update_table(table)


### Hafnia Dataset Transformations ###
class SplitsByRatios:
    def __init__(self, split_ratios: dict, seed: int = 42):
        self.split_ratios = split_ratios
        self.seed = seed

    def __call__(self, dataset: "HafniaDataset") -> "HafniaDataset":
        return splits_by_ratios(dataset, self.split_ratios, self.seed)


class ShuffleDataset:
    def __init__(self, seed: int = 42):
        self.seed = seed

    def __call__(self, dataset: "HafniaDataset") -> "HafniaDataset":
        return shuffle_dataset(dataset, self.seed)


class SampleSetBySize:
    def __init__(self, n_samples: int, seed: int = 42):
        self.n_samples = n_samples
        self.seed = seed

    def __call__(self, dataset: "HafniaDataset") -> "HafniaDataset":
        return define_sample_set_by_size(dataset, self.n_samples, self.seed)


class TransformImages:
    def __init__(self, transform: Callable[[np.ndarray], np.ndarray], path_output: Path):
        self.transform = transform
        self.path_output = path_output

    def __call__(self, dataset: "HafniaDataset") -> "HafniaDataset":
        return transform_images(dataset, self.transform, self.path_output)
