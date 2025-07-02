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
        from hafnia.dataset.builder import dataset_transformations  # noqa F401

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


def annotation_as_string(annotation: Union[type, str]) -> str:
    """Convert type annotation to string."""
    if isinstance(annotation, str):
        return annotation
    if hasattr(annotation, "__name__"):
        return annotation.__name__
    return str(annotation)


class SerializableFunction(BaseModel, Serializable, ABC):
    def __call__(self, *args) -> HafniaDataset:
        kwargs = {name: getattr(self, name) for name in self.model_fields.keys()}
        return self.get_function()(*args, **kwargs)

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


class Transforms(BaseModel, Serializable):
    loader: DatasetBuilder
    transforms: List[SerializableFunction]

    def __call__(self) -> HafniaDataset:
        """Apply all transforms to the dataset."""
        dataset = self.loader()
        for transform in self.transforms:
            dataset = transform(dataset)
        return dataset

    @field_validator("transforms", mode="plain")
    @classmethod
    def validate_transforms(cls, transforms: List[Union[Dict, SerializableFunction]]) -> List[SerializableFunction]:
        transforms_validated: List[SerializableFunction] = []
        for transform in transforms:
            if isinstance(transform, dict):
                transform = Serializable.from_dict(transform)  # type: ignore[assignment]
            if not isinstance(transform, SerializableFunction):
                raise TypeError(
                    f"All transforms must be instances of SerializableFunction, got {type(transform).__name__}."
                )
            transforms_validated.append(transform)
        return transforms_validated

    @field_serializer("transforms")
    def serialize_transforms(self, transforms: List[SerializableFunction]) -> List[dict]:
        """Serialize transforms to a list of dictionaries."""
        return [transform.model_dump() for transform in self.transforms]

    def short_name(self) -> str:
        """Return a short name for the transforms."""
        loader_name = self.loader.short_name()
        short_name_transforms = [transform.short_name() for transform in self.transforms]
        short_names = [loader_name, *short_name_transforms]
        transforms_str = ",".join(short_names)
        return f"[{transforms_str}]"


class DatasetFromName(SerializableFunction):
    name: str
    force_redownload: bool = False
    download_files: bool = True

    @staticmethod
    def get_function() -> Callable:
        return HafniaDataset.from_name

    def short_name(self) -> str:
        return self.name


class DatasetFromPath(SerializableFunction):
    path_folder: Path
    check_for_images: bool = True

    @staticmethod
    def get_function() -> Callable:
        return HafniaDataset.from_path

    def short_name(self) -> str:
        max_parent_paths_to_include = 3
        path_parts = self.path_folder.parts[-max_parent_paths_to_include:]
        short_path = "-".join(path_parts)
        return short_path


class DatasetMerger(BaseModel, Serializable):
    builders: List[DatasetBuilder]

    def __call__(self) -> HafniaDataset:
        """Concatenate datasets from all loaders."""
        if len(self.builders) == 0:
            raise ValueError("At least one loader must be provided.")

        dataset_builders = self.builders.copy()
        first_dataset_builder: DatasetBuilder = dataset_builders.pop(0)
        dataset_merged: HafniaDataset = first_dataset_builder()  # Call the first loader to get the initial dataset
        for loader in dataset_builders:  # Iterate the remaining loaders
            dataset: HafniaDataset = loader()
            dataset_merged = dataset_merged.merge(dataset)

        return dataset_merged

    def short_name(self) -> str:
        """Return a short name for the merged dataset."""
        names = [builder.short_name() for builder in self.builders]
        str_names = ",".join(names)
        return f"({str_names})"


DatasetBuilder = Union[Transforms, DatasetFromName, DatasetFromPath, DatasetMerger]
