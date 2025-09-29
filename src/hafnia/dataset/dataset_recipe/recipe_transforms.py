from typing import Callable, Dict, List, Optional, Tuple, Type, Union

from pydantic import field_validator

from hafnia.dataset.dataset_recipe.recipe_types import RecipeTransform
from hafnia.dataset.hafnia_dataset import HafniaDataset
from hafnia.dataset.primitives.primitive import Primitive


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


class ClassMapper(RecipeTransform):
    class_mapping: Union[Dict[str, str], List[Tuple[str, str]]]
    method: str = "strict"
    primitive: Optional[Type[Primitive]] = None
    task_name: Optional[str] = None

    @field_validator("class_mapping", mode="after")
    @classmethod
    def serialize_class_mapping(cls, value: Union[Dict[str, str], List[Tuple[str, str]]]) -> List[Tuple[str, str]]:
        # When stored as a recipe, dictionaries are always converted to a list of tuples
        # to preserve order in json even when stored in a postgres (jsonb) field.
        if isinstance(value, dict):
            value = list(value.items())
        return value

    @staticmethod
    def get_function() -> Callable[..., "HafniaDataset"]:
        return HafniaDataset.class_mapper


class RenameTask(RecipeTransform):
    old_task_name: str
    new_task_name: str

    @staticmethod
    def get_function() -> Callable[..., "HafniaDataset"]:
        return HafniaDataset.rename_task


class SelectSamplesByClassName(RecipeTransform):
    name: Union[List[str], str]
    task_name: Optional[str] = None
    primitive: Optional[Type[Primitive]] = None

    @staticmethod
    def get_function() -> Callable[..., "HafniaDataset"]:
        return HafniaDataset.select_samples_by_class_name
