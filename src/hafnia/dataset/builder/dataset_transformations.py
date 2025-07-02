"""
Dataset transformations to be used in the hafnia dataset specification.

Each class in this module corresponds to a dataset transformation function
defined in `operations/dataset_transformations.py`. These classes inherit from
`SerializableFunction`, which allows them to be serialized and deserialized
easily. This is particularly useful for defining dataset specifications that
can be saved, shared, and used in the TaaS (Training as a Service) platform.


Each transformation class has a static method `get_function()` that returns
the corresponding function from the `dataset_transformations` module.

The below class dataset transformations can also be used directly
in the following way:
```python
from hafnia.dataset.builder.transformations import Shuffle


shuffle_transformation == Shuffle(seed=42)
dataset: HafniaDataset = shuffle_transformation(dataset)
"""

from typing import Dict

from hafnia.dataset.builder.builders import SerializableFunction
from hafnia.dataset.operations import dataset_transformations


class SelectSamples(SerializableFunction):
    n_samples: int
    shuffle: bool = False
    seed: int = 42
    with_replacement: bool = False

    @staticmethod
    def get_function():
        return dataset_transformations.select_samples


class Shuffle(SerializableFunction):
    seed: int = 42

    @staticmethod
    def get_function():
        return dataset_transformations.shuffle


class DefineSampleSetBySize(SerializableFunction):
    n_samples: int
    seed: int = 42

    @staticmethod
    def get_function():
        return dataset_transformations.define_sample_set_by_size


class SplitIntoMultipleSplits(SerializableFunction):
    divide_split_name: str
    split_ratios: Dict[str, float]

    @staticmethod
    def get_function():
        return dataset_transformations.split_into_multiple_splits


class SplitsByRatios(SerializableFunction):
    split_ratios: Dict[str, float]
    seed: int = 42

    @staticmethod
    def get_function():
        return dataset_transformations.splits_by_ratios
