from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

from pydantic import (
    field_serializer,
    field_validator,
)

from hafnia import utils
from hafnia.dataset.dataset_helpers import dataset_name_and_version_from_string
from hafnia.dataset.dataset_names import FILENAME_RECIPE_JSON
from hafnia.dataset.dataset_recipe import recipe_transforms
from hafnia.dataset.dataset_recipe.recipe_types import (
    RecipeCreation,
    RecipeTransform,
    Serializable,
)
from hafnia.dataset.hafnia_dataset import (
    HafniaDataset,
    available_dataset_versions_from_name,
)
from hafnia.dataset.hafnia_dataset_types import DatasetMetadataFilePaths
from hafnia.dataset.primitives.primitive import Primitive
from hafnia.log import user_logger


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
    def from_name(
        name: str,
        version: Optional[str] = None,
        force_redownload: bool = False,
        download_files: bool = True,
    ) -> DatasetRecipe:
        if version == "latest":
            user_logger.info(
                f"The dataset '{name}' in a dataset recipe uses 'latest' as version. For dataset recipes the "
                "version is pinned to a specific version. Consider specifying a specific version to ensure "
                "reproducibility of your experiments. "
            )
            available_versions = available_dataset_versions_from_name(name)
            version = str(max(available_versions))
        if version is None:
            available_versions = available_dataset_versions_from_name(name)
            str_versions = ", ".join([str(v) for v in available_versions])
            raise ValueError(
                f"Version must be specified when creating a DatasetRecipe from name. "
                f"Available versions are: {str_versions}"
            )

        creation = FromName(
            name=name, version=version, force_redownload=force_redownload, download_files=download_files
        )
        return DatasetRecipe(creation=creation)

    @staticmethod
    def from_name_public_dataset(
        name: str, force_redownload: bool = False, n_samples: Optional[int] = None
    ) -> DatasetRecipe:
        creation = FromNamePublicDataset(
            name=name,
            force_redownload=force_redownload,
            n_samples=n_samples,
        )
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
    def from_recipe_field(recipe_field: Union[str, Dict[str, Any]]) -> "DatasetRecipe":
        """

        Deserialize from a recipe field which can be either a string or a dictionary.

        string: A dataset name and version string in the format 'name:version'.
        dict: A dictionary representation of the DatasetRecipe.

        """
        if isinstance(recipe_field, str):
            return DatasetRecipe.from_name_and_version_string(recipe_field)
        elif isinstance(recipe_field, dict):
            return DatasetRecipe.from_dict(recipe_field)

        raise TypeError(f"Expected str or dict for recipe_field, got {type(recipe_field).__name__}.")

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "DatasetRecipe":
        """Deserialize from a dictionary."""
        dataset_recipe = Serializable.from_dict(data)
        return dataset_recipe

    @staticmethod
    def from_recipe_id(recipe_id: str) -> "DatasetRecipe":
        """Loads a dataset recipe by id from the hafnia platform."""
        from hafnia.platform.dataset_recipe import get_dataset_recipe_by_id
        from hafnia_cli.config import Config

        cfg = Config()
        recipe_dict = get_dataset_recipe_by_id(recipe_id, cfg=cfg)
        recipe_dict = recipe_dict["template"]["body"]
        return DatasetRecipe.from_recipe_field(recipe_dict)

    @staticmethod
    def from_recipe_name(name: str) -> "DatasetRecipe":
        """Loads a dataset recipe by name from the hafnia platform"""
        from hafnia.platform.dataset_recipe import get_dataset_recipe_by_name
        from hafnia_cli.config import Config

        cfg = Config()
        recipe = get_dataset_recipe_by_name(name=name, cfg=cfg)
        if not recipe:
            raise ValueError(f"Dataset recipe '{name}' not found.")
        recipe_id = recipe["id"]
        return DatasetRecipe.from_recipe_id(recipe_id)

    @staticmethod
    def from_name_and_version_string(string: str, resolve_missing_version: bool = False) -> "DatasetRecipe":
        """
        Validates and converts a dataset name and version string (name:version) to a DatasetRecipe.from_name recipe.
        If version is missing and 'resolve_missing_version' is True, it will default to 'latest'.
        If resolve_missing_version is False, it will raise an error if version is missing.
        """

        dataset_name, version = dataset_name_and_version_from_string(
            string=string,
            resolve_missing_version=resolve_missing_version,
        )

        return DatasetRecipe.from_name(name=dataset_name, version=version)

    ### Upload, store and recipe conversions ###
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
        dict_data = self.as_dict()
        return json.dumps(dict_data, indent=indent, ensure_ascii=False)

    def as_json_file(self, path_json: Path, indent: int = 2) -> None:
        """Serialize the dataset recipe to a JSON file."""
        path_json.parent.mkdir(parents=True, exist_ok=True)
        json_str = self.as_json_str(indent=indent)
        path_json.write_text(json_str, encoding="utf-8")

    def as_dict(self) -> dict:
        """Serialize the dataset recipe to a dictionary."""
        return self.model_dump(mode="json")

    def as_platform_recipe(self, recipe_name: Optional[str], overwrite: bool = False) -> Dict:
        """Uploads dataset recipe to the hafnia platform."""
        from hafnia.platform.dataset_recipe import get_or_create_dataset_recipe
        from hafnia_cli.config import Config

        cfg = Config()

        recipe = self.as_dict()
        recipe_dict = get_or_create_dataset_recipe(recipe=recipe, name=recipe_name, overwrite=overwrite, cfg=cfg)
        return recipe_dict

    ### Dataset Recipe Transformations ###
    def shuffle(recipe: DatasetRecipe, seed: int = 42) -> DatasetRecipe:
        operation = recipe_transforms.Shuffle(seed=seed)
        recipe.append_operation(operation)
        return recipe

    def select_samples(
        recipe: DatasetRecipe,
        n_samples: int,
        shuffle: bool = True,
        seed: int = 42,
        with_replacement: bool = False,
    ) -> DatasetRecipe:
        operation = recipe_transforms.SelectSamples(
            n_samples=n_samples,
            shuffle=shuffle,
            seed=seed,
            with_replacement=with_replacement,
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

    def class_mapper(
        recipe: DatasetRecipe,
        class_mapping: Union[Dict[str, str], List[Tuple[str, str]]],
        method: str = "strict",
        primitive: Optional[Type[Primitive]] = None,
        task_name: Optional[str] = None,
    ) -> DatasetRecipe:
        operation = recipe_transforms.ClassMapper(
            class_mapping=class_mapping,
            method=method,
            primitive=primitive,
            task_name=task_name,
        )
        recipe.append_operation(operation)
        return recipe

    def rename_task(recipe: DatasetRecipe, old_task_name: str, new_task_name: str) -> DatasetRecipe:
        operation = recipe_transforms.RenameTask(old_task_name=old_task_name, new_task_name=new_task_name)
        recipe.append_operation(operation)
        return recipe

    def select_samples_by_class_name(
        recipe: DatasetRecipe,
        name: Union[List[str], str],
        task_name: Optional[str] = None,
        primitive: Optional[Type[Primitive]] = None,
    ) -> DatasetRecipe:
        operation = recipe_transforms.SelectSamplesByClassName(name=name, task_name=task_name, primitive=primitive)
        recipe.append_operation(operation)
        return recipe

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
    version: Optional[str] = None
    force_redownload: bool = False
    download_files: bool = True

    @staticmethod
    def get_function() -> Callable[..., "HafniaDataset"]:
        return HafniaDataset.from_name

    def as_short_name(self) -> str:
        return self.name

    def get_dataset_names(self) -> List[str]:
        return [self.name]


class FromNamePublicDataset(RecipeCreation):
    name: str
    force_redownload: bool = False
    n_samples: Optional[int] = None

    @staticmethod
    def get_function() -> Callable[..., "HafniaDataset"]:
        return HafniaDataset.from_name_public_dataset

    def as_short_name(self) -> str:
        return f"Torchvision('{self.name}')"

    def get_dataset_names(self) -> List[str]:
        return []


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
        names = [
            *self.recipe0.creation.get_dataset_names(),
            *self.recipe1.creation.get_dataset_names(),
        ]
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


def get_dataset_path_from_recipe(recipe: DatasetRecipe, path_datasets: Optional[Union[Path, str]] = None) -> Path:
    path_datasets = path_datasets or utils.PATH_DATASETS
    path_datasets = Path(path_datasets)
    unique_dataset_name = unique_name_from_recipe(recipe)
    return path_datasets / unique_dataset_name


def get_or_create_dataset_path_from_recipe(
    dataset_recipe: DatasetRecipe,
    force_redownload: bool = False,
    path_datasets: Optional[Union[Path, str]] = None,
) -> Path:
    path_dataset = get_dataset_path_from_recipe(dataset_recipe, path_datasets=path_datasets)

    if force_redownload:
        shutil.rmtree(path_dataset, ignore_errors=True)

    dataset_metadata_files = DatasetMetadataFilePaths.from_path(path_dataset)
    if dataset_metadata_files.exists(raise_error=False):
        return path_dataset

    path_dataset.mkdir(parents=True, exist_ok=True)
    path_recipe_json = path_dataset / FILENAME_RECIPE_JSON
    path_recipe_json.write_text(dataset_recipe.model_dump_json(indent=4))

    dataset: HafniaDataset = dataset_recipe.build()
    dataset.write(path_dataset)

    return path_dataset
