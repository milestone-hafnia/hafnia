"""
Hafnia dataset transformations that takes and returns a HafniaDataset object.

All functions here will have a corresponding function in both the HafniaDataset class
and a corresponding RecipeTransform class in the `data_recipe/recipe_transformations.py` file.

This allows each function to be used in three ways:

```python
from hafnia.dataset.operations import dataset_transformations
from hafnia.dataset.hafnia_dataset import HafniaDataset
from hafnia.dataset.data_recipe.recipe_transformations import SplitByRatios

splits_by_ratios = {"train": 0.8, "val": 0.1, "test": 0.1}

# Option 1: Using the function directly
dataset = recipe_transformations.splits_by_ratios(dataset, split_ratios=splits_by_ratios)

# Option 2: Using the method of the HafniaDataset class
dataset = dataset.splits_by_ratios(split_ratios=splits_by_ratios)

# Option 3: Using the RecipeTransform class
serializable_transform = SplitByRatios(split_ratios=splits_by_ratios)
dataset = serializable_transform(dataset)
```

Tests will ensure that all functions in this file will have a corresponding function in the
HafniaDataset class and a RecipeTransform class in the `data_recipe/recipe_transformations.py` file and
that the signatures match.
"""

from pathlib import Path
from typing import TYPE_CHECKING, Callable

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


def transform_images(
    dataset: "HafniaDataset",
    transform: Callable[[np.ndarray], np.ndarray],
    path_output: Path,
) -> "HafniaDataset":
    new_paths = []
    path_image_folder = path_output / "data"
    path_image_folder.mkdir(parents=True, exist_ok=True)

    for org_path in tqdm(dataset.samples["file_name"].to_list(), desc="Transform images"):
        org_path = Path(org_path)
        if not org_path.exists():
            raise FileNotFoundError(f"File {org_path} does not exist in the dataset.")

        image = np.array(Image.open(org_path))
        image_transformed = transform(image)
        new_path = dataset_helpers.save_image_with_hash_name(image_transformed, path_image_folder)

        if not new_path.exists():
            raise FileNotFoundError(f"Transformed file {new_path} does not exist in the dataset.")
        new_paths.append(str(new_path))

    table = dataset.samples.with_columns(pl.Series(new_paths).alias("file_name"))
    return dataset.update_table(table)
