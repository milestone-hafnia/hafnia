from __future__ import annotations

import inspect
import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable, Dict, List, Union

from pydantic import BaseModel, computed_field, field_serializer, field_validator

from hafnia.dataset.hafnia_dataset import HafniaDataset


class Serializable:
    @computed_field  # type: ignore[prop-decorator]
    @property
    def __type__(self) -> str:
        return self.__class__.__name__

    @classmethod
    def get_nested_subclasses(cls) -> List[type["Serializable"]]:
        """Recursively get all subclasses of a class."""
        from hafnia.dataset.data_recipe import recipe_transformations  # noqa F401

        all_subclasses = []
        for subclass in cls.__subclasses__():
            all_subclasses.append(subclass)
            all_subclasses.extend(subclass.get_nested_subclasses())
        return all_subclasses

    @classmethod
    def name_to_type_mapping(cls) -> dict[str, type["Serializable"]]:
        """Create a mapping from class names to class types."""
        return {subclass.__name__: subclass for subclass in cls.get_nested_subclasses()}

    @staticmethod
    def from_dict(data: dict) -> "Serializable":
        dataset_spec_args = data.copy()
        dataset_type_name = dataset_spec_args.pop("__type__", None)
        name_to_type_mapping = Serializable.name_to_type_mapping()
        SerializableClass = name_to_type_mapping[dataset_type_name]
        return SerializableClass(**dataset_spec_args)

    @staticmethod
    def from_json_str(json_str: str) -> "Serializable":
        """Deserialize from a JSON string."""
        data = json.loads(json_str)
        return Serializable.from_dict(data)

    @staticmethod
    def from_json_file(path_json: Path) -> "Serializable":
        json_str = path_json.read_text(encoding="utf-8")
        return Serializable.from_json_str(json_str)


def annotation_as_string(annotation: Union[type, str]) -> str:
    """Convert type annotation to string."""
    if isinstance(annotation, str):
        return annotation
    if hasattr(annotation, "__name__"):
        return annotation.__name__
    return str(annotation)


class RecipeTransform(BaseModel, Serializable, ABC):
    """
    This base class is used to define a recipe transform that can be serialized and deserialized.
    An important trick/feature/property of this class is that it is defined by a function that takes a HafniaDataset
    as the first argument. This allows the transform to be called like a function, while still
    being a Pydantic model that can be serialized and deserialized.

    Example of defining a RecipeTransform based on the `dataset_transformations.shuffle` function:

    ```python

    class Shuffle(RecipeTransform):
        seed: int = 42

        @staticmethod
        def get_function():
            return dataset_transformations.shuffle
    ```

    This class can then be used in a recipe as follows:
    ```python
    from hafnia.dataset.data_recipe.recipe_transformations import Shuffle
    dataset: HafniaDataset = ...  # Load or create a HafniaDataset instance
    shuffle_transform = Shuffle(seed=42)
    dataset: HafniaDataset = shuffle_transform(dataset)
    ```

    The 'check_signature' method will check if the function signature defined in `get_function()`
    matches the Pydantic model fields for all subclasses of `RecipeTransform`. In the
    above example with `Shuffle`. `Shuffle.check_signature()` will check that the
    `dataset_transformations.shuffle`-function has exactly one argument `seed` of type `int` with a default value
    of 42 that it returns a `HafniaDataset`.

    """

    def build(self, *args) -> HafniaDataset:
        kwargs = {name: getattr(self, name) for name in self.model_fields.keys()}
        return self.get_function()(*args, **kwargs)

    def __call__(self, *args) -> HafniaDataset:
        """Allow the transform to be called like a function."""
        return self.build(*args)

    @staticmethod
    @abstractmethod
    def get_function() -> Callable:
        pass

    @classmethod
    def check_signature(cls):
        sig = inspect.signature(cls.get_function())
        params = [param for param in sig.parameters.values() if param.annotation != HafniaDataset.__name__]
        HafniaDatasetName = HafniaDataset.__name__
        function_name = cls.get_function().__name__
        function_str = f"{function_name}{sig}".replace(f"'{HafniaDatasetName}'", HafniaDatasetName)
        error_msg_base = (
            f"Class definition of '{cls.__name__}' does not match the specified function '_function={function_str}'"
        )

        for param in params:
            if param.name not in cls.model_fields:
                error_msg = (
                    f"The argument '{param}' for the '{function_name}()' function is missing in the definition of '{cls.__name__}'.\n"
                    f"Action: Add '{param}'  to 'class {cls.__name__}'."
                )
                raise ValueError("\n".join([error_msg_base, error_msg]))
            model_field = cls.model_fields[param.name]
            model_field_type = annotation_as_string(model_field.annotation)
            function_param_type = annotation_as_string(param.annotation)
            if model_field_type != function_param_type:
                raise TypeError(
                    f"Type mismatch for parameter '{param.name}': expected "
                    f"{model_field_type=}, got {function_param_type=}."
                )

    def short_name(self) -> str:
        return f"{self.__class__.__name__}"


class RecipeTransforms(BaseModel, Serializable):
    recipe: DataRecipe
    transforms: List[RecipeTransform]

    def build(self) -> HafniaDataset:
        """Apply all transforms to the dataset."""
        dataset = self.recipe.build()
        for transform in self.transforms:
            dataset = transform.build(dataset)
        return dataset

    @field_validator("transforms", mode="plain")
    @classmethod
    def validate_transforms(cls, transforms: List[Union[Dict, RecipeTransform]]) -> List[RecipeTransform]:
        transforms_validated: List[RecipeTransform] = []
        for transform in transforms:
            if isinstance(transform, dict):
                transform = Serializable.from_dict(transform)  # type: ignore[assignment]
            if not isinstance(transform, RecipeTransform):
                raise TypeError(f"All transforms must be instances of RecipeTransform, got {type(transform).__name__}.")
            transforms_validated.append(transform)
        return transforms_validated

    @field_serializer("transforms")
    def serialize_transforms(self, transforms: List[RecipeTransform]) -> List[dict]:
        """Serialize transforms to a list of dictionaries."""
        return [transform.model_dump() for transform in transforms]

    def short_name(self) -> str:
        """Return a short name for the transforms."""
        recipe_name = self.recipe.short_name()
        short_name_transforms = [transform.short_name() for transform in self.transforms]
        short_names = [recipe_name, *short_name_transforms]
        transforms_str = ",".join(short_names)
        return f"Transform({transforms_str})"


class DatasetRecipeFromName(RecipeTransform):
    name: str
    force_redownload: bool = False
    download_files: bool = True

    @staticmethod
    def get_function() -> Callable:
        return HafniaDataset.from_name

    def short_name(self) -> str:
        return self.name


class DatasetRecipeFromPath(RecipeTransform):
    path_folder: Path
    check_for_images: bool = True

    @staticmethod
    def get_function() -> Callable:
        return HafniaDataset.from_path

    def short_name(self) -> str:
        max_parent_paths_to_include = 3
        path_parts = self.path_folder.parts[-max_parent_paths_to_include:]
        short_path = "|".join(path_parts)
        return f'"{short_path}"'


class RecipeMerger(BaseModel, Serializable):
    recipes: List[DataRecipe]

    def build(self) -> HafniaDataset:
        """Concatenate recipes."""
        if len(self.recipes) == 0:
            raise ValueError("At least one recipe must be provided.")

        data_recipes = self.recipes.copy()
        first_data_recipe: DataRecipe = data_recipes.pop(0)
        dataset_merged: HafniaDataset = first_data_recipe.build()  # Call the first recipe to get the initial dataset
        for recipe in data_recipes:  # Iterate the remaining recipes
            dataset: HafniaDataset = recipe.build()
            dataset_merged = dataset_merged.merge(dataset)

        return dataset_merged

    def short_name(self) -> str:
        """Return a short name for the merged dataset."""
        recipe_names = [recipe.short_name() for recipe in self.recipes]
        str_recipe_names = ",".join(recipe_names)
        return f"Merge({str_recipe_names})"


DataRecipe = Union[RecipeTransforms, DatasetRecipeFromName, DatasetRecipeFromPath, RecipeMerger]
