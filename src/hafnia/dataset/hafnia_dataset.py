from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from random import Random
from typing import Any, Dict, List, Optional, Type, Union

import more_itertools
import numpy as np
import polars as pl
import rich
from PIL import Image
from pydantic import BaseModel, field_serializer, field_validator
from rich import print as rprint
from rich.table import Table
from tqdm import tqdm

from hafnia.dataset import dataset_helpers
from hafnia.dataset.dataset_names import (
    DATASET_FILENAMES_REQUIRED,
    FILENAME_ANNOTATIONS_JSONL,
    FILENAME_ANNOTATIONS_PARQUET,
    FILENAME_DATASET_INFO,
    FILENAME_RECIPE_JSON,
    ColumnName,
    FieldName,
    SplitName,
)
from hafnia.dataset.operations import dataset_stats, dataset_transformations
from hafnia.dataset.operations.table_transformations import (
    check_image_paths,
    create_primitive_table,
    read_table_from_path,
)
from hafnia.dataset.primitives import (
    PRIMITIVE_NAME_TO_TYPE,
    PRIMITIVE_TYPES,
)
from hafnia.dataset.primitives.bbox import Bbox
from hafnia.dataset.primitives.bitmask import Bitmask
from hafnia.dataset.primitives.classification import Classification
from hafnia.dataset.primitives.polygon import Polygon
from hafnia.dataset.primitives.primitive import Primitive
from hafnia.log import user_logger


class TaskInfo(BaseModel):
    primitive: Type[Primitive]  # Primitive class or string name of the primitive, e.g. "Bbox" or "bitmask"
    class_names: Optional[List[str]]  # Class names for the tasks. To get consistent class indices specify class_names.
    name: Optional[str] = (
        None  # None to use the default primitive task name Bbox ->"bboxes", Bitmask -> "bitmasks" etc.
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
            if primitive not in PRIMITIVE_NAME_TO_TYPE:
                raise ValueError(
                    f"Primitive '{primitive}' is not recognized. Available primitives: {list(PRIMITIVE_NAME_TO_TYPE.keys())}"
                )

            return PRIMITIVE_NAME_TO_TYPE[primitive]

        if issubclass(primitive, Primitive):
            return primitive

        raise ValueError(f"Primitive must be a string or a Primitive subclass, got {type(primitive)} instead.")

    @field_serializer("primitive")
    @classmethod
    def serialize_primitive(cls, primitive: Type[Primitive]) -> str:
        if not issubclass(primitive, Primitive):
            raise ValueError(f"Primitive must be a subclass of Primitive, got {type(primitive)} instead.")
        return primitive.__name__


class DatasetInfo(BaseModel):
    dataset_name: str
    version: str
    tasks: list[TaskInfo]
    distributions: Optional[List[TaskInfo]] = None  # Distributions. TODO: FIX/REMOVE/CHANGE this
    meta: Optional[Dict[str, Any]] = None  # Metadata about the dataset, e.g. description, etc.

    def write_json(self, path: Path, indent: Optional[int] = 4) -> None:
        json_str = self.model_dump_json(indent=indent)
        path.write_text(json_str)

    @staticmethod
    def from_json_file(path: Path) -> "DatasetInfo":
        json_str = path.read_text()
        return DatasetInfo.model_validate_json(json_str)


class Sample(BaseModel):
    file_name: str
    height: int
    width: int
    split: str  # Split name, e.g., "train", "val", "test"
    is_sample: bool  # Indicates if this is a sample (True) or a metadata entry (False)
    collection_index: Optional[int] = None  # Optional e.g. frame number for video datasets
    collection_id: Optional[str] = None  # Optional e.g. video name for video datasets
    remote_path: Optional[str] = None  # Optional remote path for the image, if applicable
    sample_index: Optional[int] = None  # Don't manually set this, it is used for indexing samples in the dataset.
    classifications: Optional[List[Classification]] = None  # Optional classification primitive
    objects: Optional[List[Bbox]] = None  # List of coordinate primitives, e.g., Bbox, Bitmask, etc.
    bitmasks: Optional[List[Bitmask]] = None  # List of bitmasks, if applicable
    polygons: Optional[List[Polygon]] = None  # List of polygons, if applicable

    meta: Optional[Dict] = None  # Additional metadata, e.g., camera settings, GPS data, etc.

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
        path_image = Path(self.file_name)
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


@dataclass
class HafniaDataset:
    info: DatasetInfo
    samples: pl.DataFrame

    def __getitem__(self, item: int) -> Dict[str, Any]:
        return self.samples.row(index=item, named=True)

    def __len__(self) -> int:
        return len(self.samples)

    def __iter__(self):
        for row in self.samples.iter_rows(named=True):
            yield row

    @staticmethod
    def from_path(path_folder: Path, check_for_images: bool = True) -> "HafniaDataset":
        HafniaDataset.check_dataset_path(path_folder, raise_error=True)

        dataset_info = DatasetInfo.from_json_file(path_folder / FILENAME_DATASET_INFO)
        table = read_table_from_path(path_folder)

        # Convert from relative paths to absolute paths
        dataset_root = path_folder.absolute().as_posix() + "/"
        table = table.with_columns((dataset_root + pl.col("file_name")).alias("file_name"))
        if check_for_images:
            check_image_paths(table)
        return HafniaDataset(samples=table, info=dataset_info)

    @staticmethod
    def from_name(name: str, force_redownload: bool = False, download_files: bool = True) -> "HafniaDataset":
        """
        Load a dataset by its name. The dataset must be registered in the Hafnia platform.
        """
        from hafnia.dataset.hafnia_dataset import HafniaDataset
        from hafnia.platform.datasets import download_or_get_dataset_path

        dataset_path = download_or_get_dataset_path(
            dataset_name=name, force_redownload=force_redownload, download_files=download_files
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

        table = pl.from_records(json_samples).drop(ColumnName.SAMPLE_INDEX)
        table = table.with_row_index(name=ColumnName.SAMPLE_INDEX)  # Add sample index column

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

        path_dataset = get_or_create_dataset_path_from_recipe(dataset_recipe, path_datasets=path_datasets)
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

    # Dataset transformations
    transform_images = dataset_transformations.transform_images

    def shuffle(dataset: HafniaDataset, seed: int = 42) -> HafniaDataset:
        table = dataset.samples.sample(n=len(dataset), with_replacement=False, seed=seed, shuffle=True)
        return dataset.update_table(table)

    def select_samples(
        dataset: "HafniaDataset", n_samples: int, shuffle: bool = True, seed: int = 42, with_replacement: bool = False
    ) -> "HafniaDataset":
        if not with_replacement:
            n_samples = min(n_samples, len(dataset))
        table = dataset.samples.sample(n=n_samples, with_replacement=with_replacement, seed=seed, shuffle=shuffle)
        return dataset.update_table(table)

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
        return dataset.update_table(table)

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
        dataset_new = dataset.update_table(new_table)
        return dataset_new

    def define_sample_set_by_size(dataset: "HafniaDataset", n_samples: int, seed: int = 42) -> "HafniaDataset":
        is_sample_indices = Random(seed).sample(range(len(dataset)), n_samples)
        is_sample_column = [False for _ in range(len(dataset))]
        for idx in is_sample_indices:
            is_sample_column[idx] = True

        table = dataset.samples.with_columns(pl.Series(is_sample_column).alias("is_sample"))
        return dataset.update_table(table)

    def merge(dataset0: "HafniaDataset", dataset1: "HafniaDataset") -> "HafniaDataset":
        """
        Merges two Hafnia datasets by concatenating their samples and updating the split names.
        """
        ## Currently, only a very naive merging is implemented.
        # In the future we need to verify that the class and tasks are compatible.
        # Do they have similar classes and tasks? What to do if they don't?
        # For now, we just concatenate the samples and keep the split names as they are.
        merged_samples = pl.concat([dataset0.samples, dataset1.samples], how="vertical")
        return dataset0.update_table(merged_samples)

    # Dataset stats
    split_counts = dataset_stats.split_counts

    def as_dict_dataset_splits(self) -> Dict[str, "HafniaDataset"]:
        if ColumnName.SPLIT not in self.samples.columns:
            raise ValueError(f"Dataset must contain a '{ColumnName.SPLIT}' column.")

        splits = {}
        for split_name in SplitName.valid_splits():
            splits[split_name] = self.create_split_dataset(split_name)

        return splits

    def create_sample_dataset(self) -> "HafniaDataset":
        if ColumnName.IS_SAMPLE not in self.samples.columns:
            raise ValueError(f"Dataset must contain an '{ColumnName.IS_SAMPLE}' column.")
        table = self.samples.filter(pl.col(ColumnName.IS_SAMPLE))
        return self.update_table(table)

    def create_split_dataset(self, split_name: Union[str | List[str]]) -> "HafniaDataset":
        if isinstance(split_name, str):
            split_names = [split_name]
        elif isinstance(split_name, list):
            split_names = split_name

        for name in split_names:
            if name not in SplitName.valid_splits():
                raise ValueError(f"Invalid split name: {split_name}. Valid splits are: {SplitName.valid_splits()}")

        filtered_dataset = self.samples.filter(pl.col(ColumnName.SPLIT).is_in(split_names))
        return self.update_table(filtered_dataset)

    def get_task_by_name(self, task_name: str) -> TaskInfo:
        for task in self.info.tasks:
            if task.name == task_name:
                return task
        raise ValueError(f"Task with name {task_name} not found in dataset info.")

    def update_table(self, table: pl.DataFrame) -> "HafniaDataset":
        return HafniaDataset(info=self.info.model_copy(), samples=table)

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

    def write(self, path_folder: Path, add_version: bool = False) -> None:
        user_logger.info(f"Writing dataset to {path_folder}...")
        if not path_folder.exists():
            path_folder.mkdir(parents=True)

        new_relative_paths = []
        for org_path in tqdm(self.samples["file_name"].to_list(), desc="- Copy images"):
            new_path = dataset_helpers.copy_and_rename_file_to_hash_value(
                path_source=Path(org_path),
                path_dataset_root=path_folder,
            )
            new_relative_paths.append(str(new_path.relative_to(path_folder)))
        table = self.samples.with_columns(pl.Series(new_relative_paths).alias("file_name"))
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

    def print_stats(self) -> None:
        table_base = Table(title="Dataset Statistics", show_lines=True, box=rich.box.SIMPLE)
        table_base.add_column("Property", style="cyan")
        table_base.add_column("Value")
        table_base.add_row("Dataset Name", self.info.dataset_name)
        table_base.add_row("Version", self.info.version)
        table_base.add_row("Number of samples", str(len(self.samples)))
        rprint(table_base)
        rprint(self.info.tasks)

        splits_sets = {
            "All": SplitName.valid_splits(),
            "Train": [SplitName.TRAIN],
            "Validation": [SplitName.VAL],
            "Test": [SplitName.TEST],
        }
        rows = []
        for split_name, splits in splits_sets.items():
            dataset_split = self.create_split_dataset(splits)
            table = dataset_split.samples
            row = {}
            row["Split"] = split_name
            row["Sample "] = str(len(table))
            for PrimitiveType in PRIMITIVE_TYPES:
                column_name = PrimitiveType.column_name()
                objects_df = create_primitive_table(table, PrimitiveType=PrimitiveType, keep_sample_data=False)
                if objects_df is None:
                    continue
                for (task_name,), object_group in objects_df.group_by(FieldName.TASK_NAME):
                    count = len(object_group[FieldName.CLASS_NAME])
                    row[f"{PrimitiveType.__name__}\n{task_name}"] = str(count)
            rows.append(row)

        rich_table = Table(title="Dataset Statistics", show_lines=True, box=rich.box.SIMPLE)
        for i_row, row in enumerate(rows):
            if i_row == 0:
                for column_name in row.keys():
                    rich_table.add_column(column_name, justify="left", style="cyan")
            rich_table.add_row(*[str(value) for value in row.values()])
        rprint(rich_table)


def check_hafnia_dataset_from_path(path_dataset: Path) -> None:
    dataset = HafniaDataset.from_path(path_dataset, check_for_images=True)
    check_hafnia_dataset(dataset)


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


def check_hafnia_dataset(dataset: HafniaDataset):
    user_logger.info("Checking Hafnia dataset...")
    assert isinstance(dataset.info.version, str) and len(dataset.info.version) > 0
    assert isinstance(dataset.info.dataset_name, str) and len(dataset.info.dataset_name) > 0

    is_sample_list = set(dataset.samples.select(pl.col(ColumnName.IS_SAMPLE)).unique().to_series().to_list())
    if True not in is_sample_list:
        raise ValueError(f"The dataset should contain '{ColumnName.IS_SAMPLE}=True' samples")

    actual_splits = dataset.samples.select(pl.col(ColumnName.SPLIT)).unique().to_series().to_list()
    expected_splits = SplitName.valid_splits()
    if set(actual_splits) != set(expected_splits):
        raise ValueError(f"Expected all splits '{expected_splits}' in dataset, but got '{actual_splits}'. ")

    expected_tasks = dataset.info.tasks
    for task in expected_tasks:
        primitive = task.primitive.__name__
        column_name = task.primitive.column_name()
        primitive_column = dataset.samples[column_name]
        # msg_something_wrong = f"Something is wrong with the '{primtive_name}' task '{task.name}' in dataset '{dataset.name}'. "
        msg_something_wrong = (
            f"Something is wrong with the defined tasks ('info.tasks') in dataset '{dataset.info.dataset_name}'. \n"
            f"For '{primitive=}' and '{task.name=}' "
        )
        if primitive_column.dtype == pl.Null:
            raise ValueError(msg_something_wrong + "the column is 'Null'. Please check the dataset.")

        primitive_table = primitive_column.explode().struct.unnest().filter(pl.col(FieldName.TASK_NAME) == task.name)
        if primitive_table.is_empty():
            raise ValueError(
                msg_something_wrong
                + f"the column '{column_name}' has no {task.name=} objects. Please check the dataset."
            )

        actual_classes = set(primitive_table[FieldName.CLASS_NAME].unique().to_list())
        if task.class_names is None:
            raise ValueError(
                msg_something_wrong
                + f"the column '{column_name}' with {task.name=} has no defined classes. Please check the dataset."
            )
        defined_classes = set(task.class_names)

        if not actual_classes.issubset(defined_classes):
            raise ValueError(
                msg_something_wrong
                + f"the column '{column_name}' with {task.name=} we expected the actual classes in the dataset to \n"
                f"to be a subset of the defined classes\n\t{actual_classes=} \n\t{defined_classes=}."
            )
        # Check class_indices
        mapped_indices = primitive_table[FieldName.CLASS_NAME].map_elements(
            lambda x: task.class_names.index(x), return_dtype=pl.Int64
        )
        table_indices = primitive_table[FieldName.CLASS_IDX]

        error_msg = msg_something_wrong + (
            f"class indices in '{FieldName.CLASS_IDX}' column does not match classes ordering in 'task.class_names'"
        )
        assert mapped_indices.equals(table_indices), error_msg

    distribution = dataset.info.distributions or []
    distribution_names = [task.name for task in distribution]
    # Check that tasks found in the 'dataset.table' matches the tasks defined in 'dataset.info.tasks'
    for PrimitiveType in PRIMITIVE_TYPES:
        column_name = PrimitiveType.column_name()
        if column_name not in dataset.samples.columns:
            continue
        objects_df = create_primitive_table(dataset.samples, PrimitiveType=PrimitiveType, keep_sample_data=False)
        if objects_df is None:
            continue
        for (task_name,), object_group in objects_df.group_by(FieldName.TASK_NAME):
            has_task = any([t for t in expected_tasks if t.name == task_name and t.primitive == PrimitiveType])
            if has_task:
                continue
            if task_name in distribution_names:
                continue
            class_names = object_group[FieldName.CLASS_NAME].unique().to_list()
            raise ValueError(
                f"Task name '{task_name}' for the '{PrimitiveType.__name__}' primitive is missing in "
                f"'dataset.info.tasks' for dataset '{task_name}'. Missing task has the following "
                f"classes: {class_names}. "
            )

    for sample_dict in tqdm(dataset, desc="Checking samples in dataset"):
        sample = Sample(**sample_dict)  # Checks format of all samples with pydantic validation  # noqa: F841
