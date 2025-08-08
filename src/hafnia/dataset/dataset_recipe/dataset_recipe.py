from __future__ import annotations

import json
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Union

from pydantic import (
    field_serializer,
    field_validator,
)

from hafnia import utils
from hafnia.dataset.dataset_recipe import recipe_transforms
from hafnia.dataset.dataset_recipe.recipe_types import RecipeCreation, RecipeTransform, Serializable
from hafnia.dataset.hafnia_dataset import HafniaDataset

if TYPE_CHECKING:
    from hafnia.dataset.hafnia_dataset import HafniaDataset


class DatasetRecipe(Serializable):
    creation: RecipeCreation
    operations: Optional[List[RecipeTransform]] = None

    def build(self) -> HafniaDataset:
        dataset = self.creation.build()
        if self.operations:
            for operation in self.operations:
                dataset = operation.build(dataset)
        return dataset

    def append_operation(self, operation: RecipeTransform) -> DatasetRecipe:
        """Append an operation to the dataset recipe."""
        if self.operations is None:
            self.operations = []
        self.operations.append(operation)
        return self

    ### Creation Methods (using the 'from_X' )###
    @staticmethod
    def from_name(name: str, force_redownload: bool = False, download_files: bool = True) -> DatasetRecipe:
        creation = FromName(name=name, force_redownload=force_redownload, download_files=download_files)
        return DatasetRecipe(creation=creation)

    @staticmethod
    def from_path(path_folder: Path, check_for_images: bool = True) -> DatasetRecipe:
        creation = FromPath(path_folder=path_folder, check_for_images=check_for_images)
        return DatasetRecipe(creation=creation)

    @staticmethod
    def from_merge(recipe0: DatasetRecipe, recipe1: DatasetRecipe) -> DatasetRecipe:
        return DatasetRecipe(creation=FromMerge(recipe0=recipe0, recipe1=recipe1))

    @staticmethod
    def from_merger(recipes: List[DatasetRecipe]) -> DatasetRecipe:
        """Create a DatasetRecipe from a list of DatasetRecipes."""
        if not recipes:
            raise ValueError("The list of recipes cannot be empty.")
        if len(recipes) == 1:
            return recipes[0]
        creation = FromMerger(recipes=recipes)
        return DatasetRecipe(creation=creation)

    @staticmethod
    def from_json_str(json_str: str) -> "DatasetRecipe":
        """Deserialize from a JSON string."""
        data = json.loads(json_str)
        dataset_recipe = DatasetRecipe.from_dict(data)
        if not isinstance(dataset_recipe, DatasetRecipe):
            raise TypeError(f"Expected DatasetRecipe, got {type(dataset_recipe).__name__}.")
        return dataset_recipe

    @staticmethod
    def from_json_file(path_json: Path) -> "DatasetRecipe":
        json_str = path_json.read_text(encoding="utf-8")
        return DatasetRecipe.from_json_str(json_str)

    @staticmethod
    def from_implicit_form(recipe: Any) -> DatasetRecipe:
        """
        Recursively convert from implicit recipe to explicit form.
        Handles mixed implicit/explicit recipes.

        Conversion rules:
        - str: Will get a dataset by name -> DatasetRecipeFromName
        - Path: Will get a dataset from path -> DatasetRecipeFromPath
        - tuple: Will merge datasets specified in the tuple -> RecipeMerger
        - list: Will define a list of transformations -> RecipeTransforms

        Example: DataRecipe from dataset name:
        ```python
        recipe_implicit = "mnist"
        recipe_explicit = DatasetRecipe.from_implicit_form(recipe_implicit)
        >>> recipe_explicit
        DatasetRecipeFromName(dataset_name='mnist', force_redownload=False)
        ```

        Example: DataRecipe from tuple (merging multiple recipes):
        ```python
        recipe_implicit = ("dataset1", "dataset2")
        recipe_explicit = DatasetRecipe.from_implicit_form(recipe_implicit)
        >>> recipe_explicit
        RecipeMerger(
            recipes=[
                DatasetRecipeFromName(dataset_name='dataset1', force_redownload=False),
                DatasetRecipeFromName(dataset_name='dataset2', force_redownload=False)
            ]
        )

        Example: DataRecipe from list (recipe and transformations):
        ```python
        recipe_implicit = ["mnist", SelectSamples(n_samples=20), Shuffle(seed=123)]
        recipe_explicit = DatasetRecipe.from_implicit_form(recipe_implicit)
        >>> recipe_explicit
        Transforms(
            recipe=DatasetRecipeFromName(dataset_name='mnist', force_redownload=False),
            transforms=[SelectSamples(n_samples=20), Shuffle(seed=123)]
        )
        ```

        """
        if isinstance(recipe, DatasetRecipe):  # type: ignore
            # It is possible to do an early return if recipe is a 'DataRecipe'-type even for nested and
            # potentially mixed recipes. If you (really) think about it, this might surprise you,
            # as this will bypass the conversion logic for nested recipes.
            # However, this is not a problem as 'DataRecipe' classes are also pydantic models,
            # so if a user introduces a 'DataRecipe'-class in the recipe (in potentially
            # some nested and mixed implicit/explicit form) it will (due to pydantic validation) force
            # the user to specify all nested recipes to be converted to explicit form.
            return recipe

        if isinstance(recipe, str):  # str-type is convert to DatasetFromName
            return DatasetRecipe.from_name(name=recipe)

        if isinstance(recipe, Path):  # Path-type is convert to DatasetFromPath
            return DatasetRecipe.from_path(path_folder=recipe)

        if isinstance(recipe, tuple):  # tuple-type is convert to DatasetMerger
            recipes = [DatasetRecipe.from_implicit_form(item) for item in recipe]
            return DatasetRecipe.from_merger(recipes=recipes)

        if isinstance(recipe, list):  # list-type is convert to Transforms
            if len(recipe) == 0:
                raise ValueError("List of recipes cannot be empty")

            dataset_recipe = recipe[0]  # First element is the dataset recipe
            loader = DatasetRecipe.from_implicit_form(dataset_recipe)

            transforms = recipe[1:]  # Remaining items are transformations
            return DatasetRecipe(creation=loader.creation, operations=transforms)

        raise ValueError(f"Unsupported recipe type: {type(recipe)}")

    ### Dataset Recipe Transformations ###
    def shuffle(recipe: DatasetRecipe, seed: int = 42) -> DatasetRecipe:
        operation = recipe_transforms.Shuffle(seed=seed)
        recipe.append_operation(operation)
        return recipe

    def select_samples(
        recipe: DatasetRecipe, n_samples: int, shuffle: bool = True, seed: int = 42, with_replacement: bool = False
    ) -> DatasetRecipe:
        operation = recipe_transforms.SelectSamples(
            n_samples=n_samples, shuffle=shuffle, seed=seed, with_replacement=with_replacement
        )
        recipe.append_operation(operation)
        return recipe

    def splits_by_ratios(recipe: DatasetRecipe, split_ratios: Dict[str, float], seed: int = 42) -> DatasetRecipe:
        operation = recipe_transforms.SplitsByRatios(split_ratios=split_ratios, seed=seed)
        recipe.append_operation(operation)
        return recipe

    def split_into_multiple_splits(
        recipe: DatasetRecipe, split_name: str, split_ratios: Dict[str, float]
    ) -> DatasetRecipe:
        operation = recipe_transforms.SplitIntoMultipleSplits(split_name=split_name, split_ratios=split_ratios)
        recipe.append_operation(operation)
        return recipe

    def define_sample_set_by_size(recipe: DatasetRecipe, n_samples: int, seed: int = 42) -> DatasetRecipe:
        operation = recipe_transforms.DefineSampleSetBySize(n_samples=n_samples, seed=seed)
        recipe.append_operation(operation)
        return recipe

    ### Conversions ###
    def as_python_code(self, keep_default_fields: bool = False, as_kwargs: bool = True) -> str:
        str_operations = [self.creation.as_python_code(keep_default_fields=keep_default_fields, as_kwargs=as_kwargs)]
        if self.operations:
            for op in self.operations:
                str_operations.append(op.as_python_code(keep_default_fields=keep_default_fields, as_kwargs=as_kwargs))
        operations_str = ".".join(str_operations)
        return operations_str

    def as_short_name(self) -> str:
        """Return a short name for the transforms."""

        creation_name = self.creation.as_short_name()
        if self.operations is None or len(self.operations) == 0:
            return creation_name
        short_names = [creation_name]
        for operation in self.operations:
            short_names.append(operation.as_short_name())
        transforms_str = ",".join(short_names)
        return f"Recipe({transforms_str})"

    def as_json_str(self, indent: int = 2) -> str:
        """Serialize the dataset recipe to a JSON string."""
        data = self.model_dump(mode="json")
        # data = type_as_first_key(data)
        return json.dumps(data, indent=indent, ensure_ascii=False)

    def as_json_file(self, path_json: Path, indent: int = 2) -> None:
        """Serialize the dataset recipe to a JSON file."""
        json_str = self.as_json_str(indent=indent)
        path_json.write_text(json_str, encoding="utf-8")

    ### Helper methods ###
    def get_dataset_names(self) -> List[str]:
        """
        Get all dataset names added with 'from_name'.
        Function recursively gathers dataset names.
        """
        if self.creation is None:
            return []
        return self.creation.get_dataset_names()

    ### Validation and Serialization ###
    @field_validator("creation", mode="plain")
    @classmethod
    def validate_creation(cls, creation: Union[Dict, RecipeCreation]) -> RecipeCreation:
        if isinstance(creation, dict):
            creation = Serializable.from_dict(creation)  # type: ignore[assignment]
        if not isinstance(creation, RecipeCreation):
            raise TypeError(f"Operation must be an instance of RecipeCreation, got {type(creation).__name__}.")
        return creation

    @field_serializer("creation")
    def serialize_creation(self, creation: RecipeCreation) -> dict:
        return creation.model_dump()

    @field_validator("operations", mode="plain")
    @classmethod
    def validate_operation(cls, operations: List[Union[Dict, RecipeTransform]]) -> List[RecipeTransform]:
        if operations is None:
            return None
        validated_operations = []
        for operation in operations:
            if isinstance(operation, dict):
                operation = Serializable.from_dict(operation)  # type: ignore[assignment]
            if not isinstance(operation, RecipeTransform):
                raise TypeError(f"Operation must be an instance of RecipeTransform, got {type(operation).__name__}.")
            validated_operations.append(operation)
        return validated_operations

    @field_serializer("operations")
    def serialize_operations(self, operations: Optional[List[RecipeTransform]]) -> Optional[List[dict]]:
        """Serialize the operations to a list of dictionaries."""
        if operations is None:
            return None
        return [operation.model_dump() for operation in operations]


def unique_name_from_recipe(recipe: DatasetRecipe) -> str:
    if isinstance(recipe.creation, FromName) and recipe.operations is None:
        # If the dataset recipe is simply a DatasetFromName, we bypass the hashing logic
        # and return the name directly. The dataset is already uniquely identified by its name.
        # Add  version if need... Optionally, you may also completely delete this exception
        # and always return the unique name including the hash to support versioning.
        return recipe.creation.name  # Dataset name e.g 'mnist'
    recipe_json_str = recipe.model_dump_json()
    hash_recipe = utils.hash_from_string(recipe_json_str)
    short_recipe_str = recipe.as_short_name()
    unique_name = f"{short_recipe_str}_{hash_recipe}"
    return unique_name


def get_dataset_path_from_recipe(recipe: DatasetRecipe, path_datasets: Optional[Union[Path, str]] = None) -> Path:
    path_datasets = path_datasets or utils.PATH_DATASETS
    path_datasets = Path(path_datasets)
    unique_dataset_name = unique_name_from_recipe(recipe)
    return path_datasets / unique_dataset_name


class FromPath(RecipeCreation):
    path_folder: Path
    check_for_images: bool = True

    @staticmethod
    def get_function() -> Callable[..., "HafniaDataset"]:
        return HafniaDataset.from_path

    def as_short_name(self) -> str:
        return f"'{self.path_folder}'".replace(os.sep, "-")

    def get_dataset_names(self) -> List[str]:
        return []  # Only counts 'from_name' datasets


class FromName(RecipeCreation):
    name: str
    force_redownload: bool = False
    download_files: bool = True

    @staticmethod
    def get_function() -> Callable[..., "HafniaDataset"]:
        return HafniaDataset.from_name

    def as_short_name(self) -> str:
        return self.name

    def get_dataset_names(self) -> List[str]:
        return [self.name]


class FromMerge(RecipeCreation):
    recipe0: DatasetRecipe
    recipe1: DatasetRecipe

    @staticmethod
    def get_function():
        return HafniaDataset.merge

    def as_short_name(self) -> str:
        merger = FromMerger(recipes=[self.recipe0, self.recipe1])
        return merger.as_short_name()

    def get_dataset_names(self) -> List[str]:
        """Get the dataset names from the merged recipes."""
        names = [*self.recipe0.creation.get_dataset_names(), *self.recipe1.creation.get_dataset_names()]
        return names


class FromMerger(RecipeCreation):
    recipes: List[DatasetRecipe]

    def build(self) -> HafniaDataset:
        """Build the dataset from the merged recipes."""
        datasets = [recipe.build() for recipe in self.recipes]
        return self.get_function()(datasets=datasets)

    @staticmethod
    def get_function():
        return HafniaDataset.from_merger

    def as_short_name(self) -> str:
        return f"Merger({','.join(recipe.as_short_name() for recipe in self.recipes)})"

    def get_dataset_names(self) -> List[str]:
        """Get the dataset names from the merged recipes."""
        names = []
        for recipe in self.recipes:
            names.extend(recipe.creation.get_dataset_names())
        return names


def extract_dataset_names_from_json_dict(data: dict) -> list[str]:
    """
    Extract dataset names recursively from a JSON dictionary added with 'from_name'.

    Even if the same functionality is achieved with `DatasetRecipe.get_dataset_names()`,
    we want to keep this function in 'dipdatalib' to extract dataset names from json dictionaries
    directly.
    """
    creation_field = data.get("creation")
    if creation_field is None:
        return []
    if creation_field.get("__type__") == "FromName":
        return [creation_field["name"]]
    elif creation_field.get("__type__") == "FromMerge":
        recipe_names = ["recipe0", "recipe1"]
        dataset_name = []
        for recipe_name in recipe_names:
            recipe = creation_field.get(recipe_name)
            if recipe is None:
                continue
            dataset_name.extend(extract_dataset_names_from_json_dict(recipe))
        return dataset_name
    elif creation_field.get("__type__") == "FromMerger":
        dataset_name = []
        for recipe in creation_field.get("recipes", []):
            dataset_name.extend(extract_dataset_names_from_json_dict(recipe))
        return dataset_name
    return []
