from __future__ import annotations

import collections
import copy
import json
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from random import Random
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import more_itertools
import numpy as np
import polars as pl
from packaging.version import Version
from PIL import Image
from pydantic import BaseModel, Field, field_serializer, field_validator
from rich.progress import track

import hafnia
from hafnia.dataset import dataset_helpers
from hafnia.dataset.dataset_names import (
    DATASET_FILENAMES_REQUIRED,
    FILENAME_ANNOTATIONS_JSONL,
    FILENAME_ANNOTATIONS_PARQUET,
    FILENAME_DATASET_INFO,
    FILENAME_RECIPE_JSON,
    TAG_IS_SAMPLE,
    ColumnName,
    SplitName,
)
from hafnia.dataset.operations import (
    dataset_stats,
    dataset_transformations,
    table_transformations,
)
from hafnia.dataset.operations.table_transformations import (
    check_image_paths,
    read_samples_from_path,
)
from hafnia.dataset.primitives import PRIMITIVE_TYPES, get_primitive_type_from_string
from hafnia.dataset.primitives.bbox import Bbox
from hafnia.dataset.primitives.bitmask import Bitmask
from hafnia.dataset.primitives.classification import Classification
from hafnia.dataset.primitives.polygon import Polygon
from hafnia.dataset.primitives.primitive import Primitive
from hafnia.log import user_logger


class TaskInfo(BaseModel):
    primitive: Type[Primitive] = Field(
        description="Primitive class or string name of the primitive, e.g. 'Bbox' or 'bitmask'"
    )
    class_names: Optional[List[str]] = Field(default=None, description="Optional list of class names for the primitive")
    name: Optional[str] = Field(
        default=None,
        description=(
            "Optional name for the task. 'None' will use default name of the provided primitive. "
            "e.g. Bbox ->'bboxes', Bitmask -> 'bitmasks' etc."
        ),
    )

    def model_post_init(self, __context: Any) -> None:
        if self.name is None:
            self.name = self.primitive.default_task_name()

    # The 'primitive'-field of type 'Type[Primitive]' is not supported by pydantic out-of-the-box as
    # the 'Primitive' class is an abstract base class and for the actual primtives such as Bbox, Bitmask, Classification.
    # Below magic functions ('ensure_primitive' and 'serialize_primitive') ensures that the 'primitive' field can
    # correctly validate and serialize sub-classes (Bbox, Classification, ...).
    @field_validator("primitive", mode="plain")
    @classmethod
    def ensure_primitive(cls, primitive: Any) -> Any:
        if isinstance(primitive, str):
            return get_primitive_type_from_string(primitive)

        if issubclass(primitive, Primitive):
            return primitive

        raise ValueError(f"Primitive must be a string or a Primitive subclass, got {type(primitive)} instead.")

    @field_serializer("primitive")
    @classmethod
    def serialize_primitive(cls, primitive: Type[Primitive]) -> str:
        if not issubclass(primitive, Primitive):
            raise ValueError(f"Primitive must be a subclass of Primitive, got {type(primitive)} instead.")
        return primitive.__name__

    @field_validator("class_names", mode="after")
    @classmethod
    def validate_unique_class_names(cls, class_names: Optional[List[str]]) -> Optional[List[str]]:
        """Validate that class names are unique"""
        if class_names is None:
            return None
        duplicate_class_names = set([name for name in class_names if class_names.count(name) > 1])
        if duplicate_class_names:
            raise ValueError(
                f"Class names must be unique. The following class names appear multiple times: {duplicate_class_names}."
            )
        return class_names

    # To get unique hash value for TaskInfo objects
    def __hash__(self) -> int:
        class_names = self.class_names or []
        return hash((self.name, self.primitive.__name__, tuple(class_names)))

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, TaskInfo):
            return False
        return self.name == other.name and self.primitive == other.primitive and self.class_names == other.class_names


class DatasetInfo(BaseModel):
    dataset_name: str = Field(description="Name of the dataset, e.g. 'coco'")
    version: Optional[str] = Field(default=None, description="Version of the dataset")
    tasks: List[TaskInfo] = Field(default=None, description="List of tasks in the dataset")
    distributions: Optional[List[TaskInfo]] = Field(default=None, description="Optional list of task distributions")
    reference_bibtex: Optional[str] = Field(
        default=None,
        description="Optional, BibTeX reference to dataset publication",
    )
    reference_paper_url: Optional[str] = Field(
        default=None,
        description="Optional, URL to dataset publication",
    )
    reference_dataset_page: Optional[str] = Field(
        default=None,
        description="Optional, URL to the dataset page",
    )
    meta: Optional[Dict[str, Any]] = Field(default=None, description="Optional metadata about the dataset")
    format_version: str = Field(
        default=hafnia.__dataset_format_version__,
        description="Version of the Hafnia dataset format. You should not set this manually.",
    )
    updated_at: datetime = Field(
        default_factory=datetime.now,
        description="Timestamp of the last update to the dataset info. You should not set this manually.",
    )

    @field_validator("tasks", mode="after")
    @classmethod
    def _validate_check_for_duplicate_tasks(cls, tasks: Optional[List[TaskInfo]]) -> List[TaskInfo]:
        if tasks is None:
            return []
        task_name_counts = collections.Counter(task.name for task in tasks)
        duplicate_task_names = [name for name, count in task_name_counts.items() if count > 1]
        if duplicate_task_names:
            raise ValueError(
                f"Tasks must be unique. The following tasks appear multiple times: {duplicate_task_names}."
            )
        return tasks

    @field_validator("format_version")
    @classmethod
    def _validate_format_version(cls, format_version: str) -> str:
        try:
            Version(format_version)
        except Exception as e:
            raise ValueError(f"Invalid format_version '{format_version}'. Must be a valid version string.") from e

        if Version(format_version) > Version(hafnia.__dataset_format_version__):
            user_logger.warning(
                f"The loaded dataset format version '{format_version}' is newer than the format version "
                f"'{hafnia.__dataset_format_version__}' used in your version of Hafnia. Please consider "
                f"updating Hafnia package."
            )
        return format_version

    @field_validator("version")
    @classmethod
    def _validate_version(cls, dataset_version: Optional[str]) -> Optional[str]:
        if dataset_version is None:
            return None

        try:
            Version(dataset_version)
        except Exception as e:
            raise ValueError(f"Invalid dataset_version '{dataset_version}'. Must be a valid version string.") from e

        return dataset_version

    def check_for_duplicate_task_names(self) -> List[TaskInfo]:
        return self._validate_check_for_duplicate_tasks(self.tasks)

    def write_json(self, path: Path, indent: Optional[int] = 4) -> None:
        json_str = self.model_dump_json(indent=indent)
        path.write_text(json_str)

    @staticmethod
    def from_json_file(path: Path) -> DatasetInfo:
        json_str = path.read_text()

        # TODO: Deprecated support for old dataset info without format_version
        # Below 4 lines can be replaced by 'dataset_info = DatasetInfo.model_validate_json(json_str)'
        # when all datasets include a 'format_version' field
        json_dict = json.loads(json_str)
        if "format_version" not in json_dict:
            json_dict["format_version"] = "0.0.0"

        if "updated_at" not in json_dict:
            json_dict["updated_at"] = datetime.min.isoformat()
        dataset_info = DatasetInfo.model_validate(json_dict)

        return dataset_info

    @staticmethod
    def merge(info0: DatasetInfo, info1: DatasetInfo) -> DatasetInfo:
        """
        Merges two DatasetInfo objects into one and validates if they are compatible.
        """
        for task_ds0 in info0.tasks:
            for task_ds1 in info1.tasks:
                same_name = task_ds0.name == task_ds1.name
                same_primitive = task_ds0.primitive == task_ds1.primitive
                same_name_different_primitive = same_name and not same_primitive
                if same_name_different_primitive:
                    raise ValueError(
                        f"Cannot merge datasets with different primitives for the same task name: "
                        f"'{task_ds0.name}' has primitive '{task_ds0.primitive}' in dataset0 and "
                        f"'{task_ds1.primitive}' in dataset1."
                    )

                is_same_name_and_primitive = same_name and same_primitive
                if is_same_name_and_primitive:
                    task_ds0_class_names = task_ds0.class_names or []
                    task_ds1_class_names = task_ds1.class_names or []
                    if task_ds0_class_names != task_ds1_class_names:
                        raise ValueError(
                            f"Cannot merge datasets with different class names for the same task name and primitive: "
                            f"'{task_ds0.name}' with primitive '{task_ds0.primitive}' has class names "
                            f"{task_ds0_class_names} in dataset0 and {task_ds1_class_names} in dataset1."
                        )

        if info1.format_version != info0.format_version:
            user_logger.warning(
                "Dataset format version of the two datasets do not match. "
                f"'{info1.format_version}' vs '{info0.format_version}'."
            )
        dataset_format_version = info0.format_version
        if hafnia.__dataset_format_version__ != dataset_format_version:
            user_logger.warning(
                f"Dataset format version '{dataset_format_version}' does not match the current "
                f"Hafnia format version '{hafnia.__dataset_format_version__}'."
            )
        unique_tasks = set(info0.tasks + info1.tasks)
        distributions = set((info0.distributions or []) + (info1.distributions or []))
        meta = (info0.meta or {}).copy()
        meta.update(info1.meta or {})
        return DatasetInfo(
            dataset_name=info0.dataset_name + "+" + info1.dataset_name,
            version=None,
            tasks=list(unique_tasks),
            distributions=list(distributions),
            meta=meta,
            format_version=dataset_format_version,
        )

    def get_task_by_name(self, task_name: str) -> TaskInfo:
        """
        Get task by its name. Raises an error if the task name is not found or if multiple tasks have the same name.
        """
        tasks_with_name = [task for task in self.tasks if task.name == task_name]
        if not tasks_with_name:
            raise ValueError(f"Task with name '{task_name}' not found in dataset info.")
        if len(tasks_with_name) > 1:
            raise ValueError(f"Multiple tasks found with name '{task_name}'. This should not happen!")
        return tasks_with_name[0]

    def get_task_by_primitive(self, primitive: Union[Type[Primitive], str]) -> TaskInfo:
        """
        Get task by its primitive type. Raises an error if the primitive type is not found or if multiple tasks
        have the same primitive type.
        """
        if isinstance(primitive, str):
            primitive = get_primitive_type_from_string(primitive)

        tasks_with_primitive = [task for task in self.tasks if task.primitive == primitive]
        if not tasks_with_primitive:
            raise ValueError(f"Task with primitive {primitive} not found in dataset info.")
        if len(tasks_with_primitive) > 1:
            raise ValueError(
                f"Multiple tasks found with primitive {primitive}. Use '{self.get_task_by_name.__name__}' instead."
            )
        return tasks_with_primitive[0]

    def get_task_by_task_name_and_primitive(
        self,
        task_name: Optional[str],
        primitive: Optional[Union[Type[Primitive], str]],
    ) -> TaskInfo:
        """
        Logic to get a unique task based on the provided 'task_name' and/or 'primitive'.
        If both 'task_name' and 'primitive' are None, the dataset must have only one task.
        """
        task = dataset_transformations.get_task_info_from_task_name_and_primitive(
            tasks=self.tasks,
            primitive=primitive,
            task_name=task_name,
        )
        return task

    def replace_task(self, old_task: TaskInfo, new_task: Optional[TaskInfo]) -> DatasetInfo:
        dataset_info = self.model_copy(deep=True)
        has_task = any(t for t in dataset_info.tasks if t.name == old_task.name and t.primitive == old_task.primitive)
        if not has_task:
            raise ValueError(f"Task '{old_task.__repr__()}' not found in dataset info.")

        new_tasks = []
        for task in dataset_info.tasks:
            if task.name == old_task.name and task.primitive == old_task.primitive:
                if new_task is None:
                    continue  # Remove the task
                new_tasks.append(new_task)
            else:
                new_tasks.append(task)

        dataset_info.tasks = new_tasks
        return dataset_info


class Sample(BaseModel):
    file_path: str = Field(description="Path to the image file")
    height: int = Field(description="Height of the image")
    width: int = Field(description="Width of the image")
    split: str = Field(description="Split name, e.g., 'train', 'val', 'test'")
    tags: List[str] = Field(
        default_factory=list,
        description="Tags for a given sample. Used for creating subsets of the dataset.",
    )
    collection_index: Optional[int] = Field(default=None, description="Optional e.g. frame number for video datasets")
    collection_id: Optional[str] = Field(default=None, description="Optional e.g. video name for video datasets")
    remote_path: Optional[str] = Field(default=None, description="Optional remote path for the image, if applicable")
    sample_index: Optional[int] = Field(
        default=None,
        description="Don't manually set this, it is used for indexing samples in the dataset.",
    )
    classifications: Optional[List[Classification]] = Field(
        default=None, description="Optional list of classifications"
    )
    objects: Optional[List[Bbox]] = Field(default=None, description="Optional list of objects (bounding boxes)")
    bitmasks: Optional[List[Bitmask]] = Field(default=None, description="Optional list of bitmasks")
    polygons: Optional[List[Polygon]] = Field(default=None, description="Optional list of polygons")

    attribution: Optional[Attribution] = Field(default=None, description="Attribution information for the image")
    dataset_name: Optional[str] = Field(
        default=None,
        description=(
            "Don't manually set this, it will be automatically defined during initialization. "
            "Name of the dataset the sample belongs to. E.g. 'coco-2017' or 'midwest-vehicle-detection'."
        ),
    )
    meta: Optional[Dict] = Field(
        default=None,
        description="Additional metadata, e.g., camera settings, GPS data, etc.",
    )

    def get_annotations(self, primitive_types: Optional[List[Type[Primitive]]] = None) -> List[Primitive]:
        """
        Returns a list of all annotations (classifications, objects, bitmasks, polygons) for the sample.
        """
        primitive_types = primitive_types or PRIMITIVE_TYPES
        annotations_primitives = [
            getattr(self, primitive_type.column_name(), None) for primitive_type in primitive_types
        ]
        annotations = more_itertools.flatten(
            [primitives for primitives in annotations_primitives if primitives is not None]
        )

        return list(annotations)

    def read_image_pillow(self) -> Image.Image:
        """
        Reads the image from the file path and returns it as a PIL Image.
        Raises FileNotFoundError if the image file does not exist.
        """
        path_image = Path(self.file_path)
        if not path_image.exists():
            raise FileNotFoundError(f"Image file {path_image} does not exist. Please check the file path.")

        image = Image.open(str(path_image))
        return image

    def read_image(self) -> np.ndarray:
        image_pil = self.read_image_pillow()
        image = np.array(image_pil)
        return image

    def draw_annotations(self, image: Optional[np.ndarray] = None) -> np.ndarray:
        from hafnia.visualizations import image_visualizations

        image = image or self.read_image()
        annotations = self.get_annotations()
        annotations_visualized = image_visualizations.draw_annotations(image=image, primitives=annotations)
        return annotations_visualized


class License(BaseModel):
    """License information"""

    name: Optional[str] = Field(
        default=None,
        description="License name. E.g. 'Creative Commons: Attribution 2.0 Generic'",
        max_length=100,
    )
    name_short: Optional[str] = Field(
        default=None,
        description="License short name or abbreviation. E.g. 'CC BY 4.0'",
        max_length=100,
    )
    url: Optional[str] = Field(
        default=None,
        description="License URL e.g. https://creativecommons.org/licenses/by/4.0/",
    )
    description: Optional[str] = Field(
        default=None,
        description=(
            "License description e.g. 'You must give appropriate credit, provide a "
            "link to the license, and indicate if changes were made.'"
        ),
    )

    valid_date: Optional[datetime] = Field(
        default=None,
        description="License valid date. E.g. '2023-01-01T00:00:00Z'",
    )

    permissions: Optional[List[str]] = Field(
        default=None,
        description="License permissions. Allowed to Access, Label, Distribute, Represent and Modify data.",
    )
    liability: Optional[str] = Field(
        default=None,
        description="License liability. Optional and not always applicable.",
    )
    location: Optional[str] = Field(
        default=None,
        description=(
            "License Location. E.g. Iowa state. This is essential to understand the industry and "
            "privacy location specific rules that applies to the data. Optional and not always applicable."
        ),
    )
    notes: Optional[str] = Field(
        default=None,
        description="Additional license notes. Optional and not always applicable.",
    )


class Attribution(BaseModel):
    """Attribution information for the image: Giving source and credit to the original creator"""

    title: Optional[str] = Field(default=None, description="Title of the image", max_length=255)
    creator: Optional[str] = Field(default=None, description="Creator of the image", max_length=255)
    creator_url: Optional[str] = Field(default=None, description="URL of the creator", max_length=255)
    date_captured: Optional[datetime] = Field(default=None, description="Date when the image was captured")
    copyright_notice: Optional[str] = Field(default=None, description="Copyright notice for the image", max_length=255)
    licenses: Optional[List[License]] = Field(default=None, description="List of licenses for the image")
    disclaimer: Optional[str] = Field(default=None, description="Disclaimer for the image", max_length=255)
    changes: Optional[str] = Field(default=None, description="Changes made to the image", max_length=255)
    source_url: Optional[str] = Field(default=None, description="Source URL for the image", max_length=255)


@dataclass
class HafniaDataset:
    info: DatasetInfo
    samples: pl.DataFrame

    # Function mapping: Dataset stats
    split_counts = dataset_stats.split_counts
    class_counts_for_task = dataset_stats.class_counts_for_task
    class_counts_all = dataset_stats.class_counts_all

    # Function mapping: Print stats
    print_stats = dataset_stats.print_stats
    print_sample_and_task_counts = dataset_stats.print_sample_and_task_counts
    print_class_distribution = dataset_stats.print_class_distribution

    # Function mapping: Dataset checks
    check_dataset = dataset_stats.check_dataset
    check_dataset_tasks = dataset_stats.check_dataset_tasks

    # Function mapping: Dataset transformations
    transform_images = dataset_transformations.transform_images

    def __getitem__(self, item: int) -> Dict[str, Any]:
        return self.samples.row(index=item, named=True)

    def __len__(self) -> int:
        return len(self.samples)

    def __iter__(self):
        for row in self.samples.iter_rows(named=True):
            yield row

    def __post_init__(self):
        self.samples, self.info = _dataset_corrections(self.samples, self.info)

    @staticmethod
    def from_path(path_folder: Path, check_for_images: bool = True) -> "HafniaDataset":
        path_folder = Path(path_folder)
        HafniaDataset.check_dataset_path(path_folder, raise_error=True)

        dataset_info = DatasetInfo.from_json_file(path_folder / FILENAME_DATASET_INFO)
        samples = read_samples_from_path(path_folder)
        samples, dataset_info = _dataset_corrections(samples, dataset_info)

        # Convert from relative paths to absolute paths
        dataset_root = path_folder.absolute().as_posix() + "/"
        samples = samples.with_columns((dataset_root + pl.col(ColumnName.FILE_PATH)).alias(ColumnName.FILE_PATH))
        if check_for_images:
            check_image_paths(samples)
        return HafniaDataset(samples=samples, info=dataset_info)

    @staticmethod
    def from_name(name: str, force_redownload: bool = False, download_files: bool = True) -> "HafniaDataset":
        """
        Load a dataset by its name. The dataset must be registered in the Hafnia platform.
        """
        from hafnia.platform.datasets import download_or_get_dataset_path

        dataset_path = download_or_get_dataset_path(
            dataset_name=name,
            force_redownload=force_redownload,
            download_files=download_files,
        )
        return HafniaDataset.from_path(dataset_path, check_for_images=download_files)

    @staticmethod
    def from_samples_list(samples_list: List, info: DatasetInfo) -> "HafniaDataset":
        sample = samples_list[0]
        if isinstance(sample, Sample):
            json_samples = [sample.model_dump(mode="json") for sample in samples_list]
        elif isinstance(sample, dict):
            json_samples = samples_list
        else:
            raise TypeError(f"Unsupported sample type: {type(sample)}. Expected Sample or dict.")

        table = pl.from_records(json_samples)
        table = table.drop(ColumnName.SAMPLE_INDEX).with_row_index(name=ColumnName.SAMPLE_INDEX)

        # Add 'dataset_name' to samples
        table = table.with_columns(
            pl.when(pl.col(ColumnName.DATASET_NAME).is_null())
            .then(pl.lit(info.dataset_name))
            .otherwise(pl.col(ColumnName.DATASET_NAME))
            .alias(ColumnName.DATASET_NAME)
        )
        return HafniaDataset(info=info, samples=table)

    @staticmethod
    def from_recipe(dataset_recipe: Any) -> "HafniaDataset":
        """
        Load a dataset from a recipe. The recipe can be a string (name of the dataset), a dictionary, or a DataRecipe object.
        """
        from hafnia.dataset.dataset_recipe.dataset_recipe import DatasetRecipe

        recipe_explicit = DatasetRecipe.from_implicit_form(dataset_recipe)

        return recipe_explicit.build()  # Build dataset from the recipe

    @staticmethod
    def from_merge(dataset0: "HafniaDataset", dataset1: "HafniaDataset") -> "HafniaDataset":
        return HafniaDataset.merge(dataset0, dataset1)

    @staticmethod
    def from_recipe_with_cache(
        dataset_recipe: Any,
        force_redownload: bool = False,
        path_datasets: Optional[Union[Path, str]] = None,
    ) -> "HafniaDataset":
        """
        Loads a dataset from a recipe and caches it to disk.
        If the dataset is already cached, it will be loaded from the cache.
        """

        path_dataset = get_or_create_dataset_path_from_recipe(
            dataset_recipe,
            path_datasets=path_datasets,
            force_redownload=force_redownload,
        )
        return HafniaDataset.from_path(path_dataset, check_for_images=False)

    @staticmethod
    def from_merger(
        datasets: List[HafniaDataset],
    ) -> "HafniaDataset":
        """
        Merges multiple Hafnia datasets into one.
        """
        if len(datasets) == 0:
            raise ValueError("No datasets to merge. Please provide at least one dataset.")

        if len(datasets) == 1:
            return datasets[0]

        merged_dataset = datasets[0]
        remaining_datasets = datasets[1:]
        for dataset in remaining_datasets:
            merged_dataset = HafniaDataset.merge(merged_dataset, dataset)
        return merged_dataset

    @staticmethod
    def from_name_public_dataset(
        name: str,
        force_redownload: bool = False,
        n_samples: Optional[int] = None,
    ) -> HafniaDataset:
        from hafnia.dataset.format_conversions.torchvision_datasets import (
            torchvision_to_hafnia_converters,
        )

        name_to_torchvision_function = torchvision_to_hafnia_converters()

        if name not in name_to_torchvision_function:
            raise ValueError(
                f"Unknown torchvision dataset name: {name}. Supported: {list(name_to_torchvision_function.keys())}"
            )
        vision_dataset = name_to_torchvision_function[name]
        return vision_dataset(
            force_redownload=force_redownload,
            n_samples=n_samples,
        )

    def shuffle(dataset: HafniaDataset, seed: int = 42) -> HafniaDataset:
        table = dataset.samples.sample(n=len(dataset), with_replacement=False, seed=seed, shuffle=True)
        return dataset.update_samples(table)

    def select_samples(
        dataset: "HafniaDataset",
        n_samples: int,
        shuffle: bool = True,
        seed: int = 42,
        with_replacement: bool = False,
    ) -> "HafniaDataset":
        """
        Create a new dataset with a subset of samples.
        """
        if not with_replacement:
            n_samples = min(n_samples, len(dataset))
        table = dataset.samples.sample(n=n_samples, with_replacement=with_replacement, seed=seed, shuffle=shuffle)
        return dataset.update_samples(table)

    def splits_by_ratios(dataset: "HafniaDataset", split_ratios: Dict[str, float], seed: int = 42) -> "HafniaDataset":
        """
        Divides the dataset into splits based on the provided ratios.

        Example: Defining split ratios and applying the transformation

        >>> dataset = HafniaDataset.read_from_path(Path("path/to/dataset"))
        >>> split_ratios = {SplitName.TRAIN: 0.8, SplitName.VAL: 0.1, SplitName.TEST: 0.1}
        >>> dataset_with_splits = splits_by_ratios(dataset, split_ratios, seed=42)
        Or use the function as a
        >>> dataset_with_splits = dataset.splits_by_ratios(split_ratios, seed=42)
        """
        n_items = len(dataset)
        split_name_column = dataset_helpers.create_split_name_list_from_ratios(
            split_ratios=split_ratios, n_items=n_items, seed=seed
        )
        table = dataset.samples.with_columns(pl.Series(split_name_column).alias("split"))
        return dataset.update_samples(table)

    def split_into_multiple_splits(
        dataset: "HafniaDataset",
        split_name: str,
        split_ratios: Dict[str, float],
    ) -> "HafniaDataset":
        """
        Divides a dataset split ('split_name') into multiple splits based on the provided split
        ratios ('split_ratios'). This is especially useful for some open datasets where they have only provide
        two splits or only provide annotations for two splits. This function allows you to create additional
        splits based on the provided ratios.

        Example: Defining split ratios and applying the transformation
        >>> dataset = HafniaDataset.read_from_path(Path("path/to/dataset"))
        >>> split_name = SplitName.TEST
        >>> split_ratios = {SplitName.TEST: 0.8, SplitName.VAL: 0.2}
        >>> dataset_with_splits = split_into_multiple_splits(dataset, split_name, split_ratios)
        """
        dataset_split_to_be_divided = dataset.create_split_dataset(split_name=split_name)
        if len(dataset_split_to_be_divided) == 0:
            split_counts = dict(dataset.samples.select(pl.col(ColumnName.SPLIT).value_counts()).iter_rows())
            raise ValueError(f"No samples in the '{split_name}' split to divide into multiple splits. {split_counts=}")
        assert len(dataset_split_to_be_divided) > 0, f"No samples in the '{split_name}' split!"
        dataset_split_to_be_divided = dataset_split_to_be_divided.splits_by_ratios(split_ratios=split_ratios, seed=42)

        remaining_data = dataset.samples.filter(pl.col(ColumnName.SPLIT).is_in([split_name]).not_())
        new_table = pl.concat([remaining_data, dataset_split_to_be_divided.samples], how="vertical")
        dataset_new = dataset.update_samples(new_table)
        return dataset_new

    def define_sample_set_by_size(dataset: "HafniaDataset", n_samples: int, seed: int = 42) -> "HafniaDataset":
        """
        Defines a sample set randomly by selecting 'n_samples' samples from the dataset.
        """
        samples = dataset.samples

        # Remove any pre-existing "sample"-tags
        samples = samples.with_columns(
            pl.col(ColumnName.TAGS).list.eval(pl.element().filter(pl.element() != TAG_IS_SAMPLE)).alias(ColumnName.TAGS)
        )

        # Add "sample" to tags column for the selected samples
        is_sample_indices = Random(seed).sample(range(len(dataset)), n_samples)
        samples = samples.with_columns(
            pl.when(pl.int_range(len(samples)).is_in(is_sample_indices))
            .then(pl.col(ColumnName.TAGS).list.concat(pl.lit([TAG_IS_SAMPLE])))
            .otherwise(pl.col(ColumnName.TAGS))
        )
        return dataset.update_samples(samples)

    def class_mapper(
        dataset: "HafniaDataset",
        class_mapping: Union[Dict[str, str], List[Tuple[str, str]]],
        method: str = "strict",
        primitive: Optional[Type[Primitive]] = None,
        task_name: Optional[str] = None,
    ) -> "HafniaDataset":
        """
        Map class names to new class names using a strict mapping.
        A strict mapping means that all class names in the dataset must be mapped to a new class name.
        If a class name is not mapped, an error is raised.

        The class indices are determined by the order of appearance of the new class names in the mapping.
        Duplicates in the new class names are removed, preserving the order of first appearance.

        E.g.

        mnist = HafniaDataset.from_name("mnist")
        strict_class_mapping = {
            "1 - one": "odd",   # 'odd' appears first and becomes class index 0
            "3 - three": "odd",
            "5 - five": "odd",
            "7 - seven": "odd",
            "9 - nine": "odd",
            "0 - zero": "even",  # 'even' appears second and becomes class index 1
            "2 - two": "even",
            "4 - four": "even",
            "6 - six": "even",
            "8 - eight": "even",
        }

        dataset_new = class_mapper(dataset=mnist, class_mapping=strict_class_mapping)

        """
        return dataset_transformations.class_mapper(
            dataset=dataset,
            class_mapping=class_mapping,
            method=method,
            primitive=primitive,
            task_name=task_name,
        )

    def rename_task(
        dataset: "HafniaDataset",
        old_task_name: str,
        new_task_name: str,
    ) -> "HafniaDataset":
        """
        Rename a task in the dataset.
        """
        return dataset_transformations.rename_task(
            dataset=dataset, old_task_name=old_task_name, new_task_name=new_task_name
        )

    def select_samples_by_class_name(
        dataset: HafniaDataset,
        name: Union[List[str], str],
        task_name: Optional[str] = None,
        primitive: Optional[Type[Primitive]] = None,
    ) -> HafniaDataset:
        """
        Select samples that contain at least one annotation with the specified class name(s).
        If 'task_name' and 'primitive' are not provided, the function will attempt to infer the task.
        """
        return dataset_transformations.select_samples_by_class_name(
            dataset=dataset, name=name, task_name=task_name, primitive=primitive
        )

    def merge(dataset0: "HafniaDataset", dataset1: "HafniaDataset") -> "HafniaDataset":
        """
        Merges two Hafnia datasets by concatenating their samples and updating the split names.
        """

        # Merges dataset info and checks for compatibility
        merged_info = DatasetInfo.merge(dataset0.info, dataset1.info)

        # Merges samples tables (removes incompatible columns)
        merged_samples = table_transformations.merge_samples(samples0=dataset0.samples, samples1=dataset1.samples)

        # Check if primitives have been removed during the merge_samples
        for task in copy.deepcopy(merged_info.tasks):
            if task.primitive.column_name() not in merged_samples.columns:
                user_logger.warning(
                    f"Task '{task.name}' with primitive '{task.primitive.__name__}' has been removed during the merge. "
                    "This happens if the two datasets do not have the same primitives."
                )
                merged_info = merged_info.replace_task(old_task=task, new_task=None)

        return HafniaDataset(info=merged_info, samples=merged_samples)

    def as_dict_dataset_splits(self) -> Dict[str, "HafniaDataset"]:
        """
        Splits the dataset into multiple datasets based on the 'split' column.
        Returns a dictionary with split names as keys and HafniaDataset objects as values.
        """
        if ColumnName.SPLIT not in self.samples.columns:
            raise ValueError(f"Dataset must contain a '{ColumnName.SPLIT}' column.")

        splits = {}
        for split_name in SplitName.valid_splits():
            splits[split_name] = self.create_split_dataset(split_name)

        return splits

    def create_sample_dataset(self) -> "HafniaDataset":
        # Backwards compatibility. Remove in future versions when dataset have been updated
        if "is_sample" in self.samples.columns:
            user_logger.warning(
                "'is_sample' column found in the dataset. This column is deprecated and will be removed in future versions. "
                "Please use the 'tags' column with the tag 'sample' instead."
            )
            table = self.samples.filter(pl.col("is_sample") == True)  # noqa: E712
            return self.update_samples(table)

        if ColumnName.TAGS not in self.samples.columns:
            raise ValueError(f"Dataset must contain an '{ColumnName.TAGS}' column.")

        table = self.samples.filter(
            pl.col(ColumnName.TAGS).list.eval(pl.element().filter(pl.element() == TAG_IS_SAMPLE)).list.len() > 0
        )
        return self.update_samples(table)

    def create_split_dataset(self, split_name: Union[str | List[str]]) -> "HafniaDataset":
        if isinstance(split_name, str):
            split_names = [split_name]
        elif isinstance(split_name, list):
            split_names = split_name

        for name in split_names:
            if name not in SplitName.valid_splits():
                raise ValueError(f"Invalid split name: {split_name}. Valid splits are: {SplitName.valid_splits()}")

        filtered_dataset = self.samples.filter(pl.col(ColumnName.SPLIT).is_in(split_names))
        return self.update_samples(filtered_dataset)

    def update_samples(self, table: pl.DataFrame) -> "HafniaDataset":
        dataset = HafniaDataset(info=self.info.model_copy(deep=True), samples=table)
        dataset.check_dataset_tasks()
        return dataset

    @staticmethod
    def check_dataset_path(path_dataset: Path, raise_error: bool = True) -> bool:
        """
        Checks if the dataset path exists and contains the required files.
        Returns True if the dataset is valid, otherwise raises an error or returns False.
        """
        if not path_dataset.exists():
            if raise_error:
                raise FileNotFoundError(f"Dataset path {path_dataset} does not exist.")
            return False

        required_files = [
            FILENAME_DATASET_INFO,
            FILENAME_ANNOTATIONS_JSONL,
            FILENAME_ANNOTATIONS_PARQUET,
        ]
        for filename in required_files:
            if not (path_dataset / filename).exists():
                if raise_error:
                    raise FileNotFoundError(f"Required file {filename} not found in {path_dataset}.")
                return False

        return True

    def copy(self) -> "HafniaDataset":
        return HafniaDataset(info=self.info.model_copy(deep=True), samples=self.samples.clone())

    def write(self, path_folder: Path, add_version: bool = False, drop_null_cols: bool = True) -> None:
        user_logger.info(f"Writing dataset to {path_folder}...")
        if not path_folder.exists():
            path_folder.mkdir(parents=True)

        new_relative_paths = []
        org_paths = self.samples[ColumnName.FILE_PATH].to_list()
        for org_path in track(org_paths, description="- Copy images"):
            new_path = dataset_helpers.copy_and_rename_file_to_hash_value(
                path_source=Path(org_path),
                path_dataset_root=path_folder,
            )
            new_relative_paths.append(str(new_path.relative_to(path_folder)))
        table = self.samples.with_columns(pl.Series(new_relative_paths).alias(ColumnName.FILE_PATH))

        if drop_null_cols:  # Drops all unused/Null columns
            table = table.drop(pl.selectors.by_dtype(pl.Null))

        table.write_ndjson(path_folder / FILENAME_ANNOTATIONS_JSONL)  # Json for readability
        table.write_parquet(path_folder / FILENAME_ANNOTATIONS_PARQUET)  # Parquet for speed
        self.info.write_json(path_folder / FILENAME_DATASET_INFO)

        if add_version:
            path_version = path_folder / "versions" / f"{self.info.version}"
            path_version.mkdir(parents=True, exist_ok=True)
            for filename in DATASET_FILENAMES_REQUIRED:
                shutil.copy2(path_folder / filename, path_version / filename)

    def __eq__(self, value) -> bool:
        if not isinstance(value, HafniaDataset):
            return False

        if self.info != value.info:
            return False

        if not isinstance(self.samples, pl.DataFrame) or not isinstance(value.samples, pl.DataFrame):
            return False

        if not self.samples.equals(value.samples):
            return False
        return True


def check_hafnia_dataset_from_path(path_dataset: Path) -> None:
    dataset = HafniaDataset.from_path(path_dataset, check_for_images=True)
    dataset.check_dataset()


def get_or_create_dataset_path_from_recipe(
    dataset_recipe: Any,
    force_redownload: bool = False,
    path_datasets: Optional[Union[Path, str]] = None,
) -> Path:
    from hafnia.dataset.dataset_recipe.dataset_recipe import (
        DatasetRecipe,
        get_dataset_path_from_recipe,
    )

    recipe: DatasetRecipe = DatasetRecipe.from_implicit_form(dataset_recipe)
    path_dataset = get_dataset_path_from_recipe(recipe, path_datasets=path_datasets)

    if force_redownload:
        shutil.rmtree(path_dataset, ignore_errors=True)

    if HafniaDataset.check_dataset_path(path_dataset, raise_error=False):
        return path_dataset

    path_dataset.mkdir(parents=True, exist_ok=True)
    path_recipe_json = path_dataset / FILENAME_RECIPE_JSON
    path_recipe_json.write_text(recipe.model_dump_json(indent=4))

    dataset: HafniaDataset = recipe.build()
    dataset.write(path_dataset)

    return path_dataset


def _dataset_corrections(samples: pl.DataFrame, dataset_info: DatasetInfo) -> Tuple[pl.DataFrame, DatasetInfo]:
    format_version_of_dataset = Version(dataset_info.format_version)

    ## Backwards compatibility fixes for older dataset versions
    if format_version_of_dataset <= Version("0.3.0"):
        if ColumnName.DATASET_NAME not in samples.columns:
            samples = samples.with_columns(pl.lit(dataset_info.dataset_name).alias(ColumnName.DATASET_NAME))

        if "file_name" in samples.columns:
            samples = samples.rename({"file_name": ColumnName.FILE_PATH})

        if ColumnName.SAMPLE_INDEX not in samples.columns:
            samples = samples.with_row_index(name=ColumnName.SAMPLE_INDEX)

        # Backwards compatibility: If tags-column doesn't exist, create it with empty lists
        if ColumnName.TAGS not in samples.columns:
            tags_column: List[List[str]] = [[] for _ in range(len(samples))]  # type: ignore[annotation-unchecked]
            samples = samples.with_columns(pl.Series(tags_column, dtype=pl.List(pl.String)).alias(ColumnName.TAGS))

    return samples, dataset_info
