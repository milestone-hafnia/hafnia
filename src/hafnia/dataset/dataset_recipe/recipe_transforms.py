from typing import TYPE_CHECKING, Callable, Dict

from hafnia.dataset.dataset_recipe.recipe_types import RecipeTransform
from hafnia.dataset.hafnia_dataset import HafniaDataset

if TYPE_CHECKING:
    pass


class Shuffle(RecipeTransform):
    seed: int = 42

    @staticmethod
    def get_function() -> Callable[..., "HafniaDataset"]:
        return HafniaDataset.shuffle


class SelectSamples(RecipeTransform):
    n_samples: int
    shuffle: bool = True
    seed: int = 42
    with_replacement: bool = False

    @staticmethod
    def get_function() -> Callable[..., "HafniaDataset"]:
        return HafniaDataset.select_samples


class SplitsByRatios(RecipeTransform):
    split_ratios: Dict[str, float]
    seed: int = 42

    @staticmethod
    def get_function() -> Callable[..., "HafniaDataset"]:
        return HafniaDataset.splits_by_ratios


class SplitIntoMultipleSplits(RecipeTransform):
    split_name: str
    split_ratios: Dict[str, float]

    @staticmethod
    def get_function() -> Callable[..., "HafniaDataset"]:
        return HafniaDataset.split_into_multiple_splits


class DefineSampleSetBySize(RecipeTransform):
    n_samples: int
    seed: int = 42

    @staticmethod
    def get_function() -> Callable[..., "HafniaDataset"]:
        return HafniaDataset.define_sample_set_by_size
