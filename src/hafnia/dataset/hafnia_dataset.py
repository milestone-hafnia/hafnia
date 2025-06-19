from __future__ import annotations

import os
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, Union

import more_itertools
import numpy as np
import polars as pl
import rich
from PIL import Image
from pydantic import BaseModel, field_serializer, field_validator
from rich.table import Table
from tqdm import tqdm

from hafnia.dataset import dataset_transformation
from hafnia.dataset.base_types import Primitive
from hafnia.dataset.dataset_names import (
    FILENAME_ANNOTATIONS_JSONL,
    FILENAME_ANNOTATIONS_PARQUET,
    FILENAME_DATASET_INFO,
    ColumnName,
    FieldName,
    SplitName,
)
from hafnia.dataset.shape_primitives import (
    PRIMITIVE_NAME_TO_TYPE,
    PRIMITIVE_TYPES,
    Bbox,
    Bitmask,
    Classification,
    Polygon,
)
from hafnia.dataset.table_transformations import (
    check_image_paths,
    create_primitive_table,
    read_table_from_path,
)


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
    image_id: str
    file_name: str
    height: int
    width: int
    split: str  # Split name, e.g., "train", "val", "test"
    is_sample: bool  # Indicates if this is a sample (True) or a metadata entry (False)
    frame_number: Optional[int] = None  # Optional frame number for video datasets
    video_name: Optional[str] = None  # Optional video name for video datasets
    remote_path: Optional[str] = None  # Optional remote path for the image, if applicable
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
        from hafnia.dataset import image_operations

        image = image or self.read_image()
        annotations = self.get_annotations()
        annotations_visualized = image_operations.draw_annotations(image=image, primitives=annotations)
        return annotations_visualized


@dataclass
class HafniaDataset:
    info: DatasetInfo
    table: pl.DataFrame

    def __getitem__(self, item: int) -> Dict[str, Any]:
        return self.table.row(index=item, named=True)

    def __len__(self) -> int:
        return len(self.table)

    def __iter__(self):
        for row in self.table.iter_rows(named=True):
            yield row

    # Dataset transformations
    apply_image_transform = dataset_transformation.transform_images
    sample = dataset_transformation.sample
    shuffle = dataset_transformation.shuffle_dataset
    split_by_ratios = dataset_transformation.splits_by_ratios
    divide_split_into_multiple_splits = dataset_transformation.divide_split_into_multiple_splits
    sample_set_by_size = dataset_transformation.define_sample_set_by_size

    @staticmethod
    def from_samples(samples: List, info: DatasetInfo) -> "HafniaDataset":
        sample = samples[0]
        if isinstance(sample, Sample):
            json_samples = [sample.model_dump(mode="json") for sample in samples]
        elif isinstance(sample, dict):
            json_samples = samples
        else:
            raise TypeError(f"Unsupported sample type: {type(sample)}. Expected Sample or dict.")

        table = pl.from_records(json_samples)

        has_unique_image_ids = table.select(pl.col("image_id").is_unique().all()).item()
        if not has_unique_image_ids:
            raise ValueError("Dataset contains non-unique image IDs. Please ensure all image IDs are unique.")
        return HafniaDataset(info=info, table=table)

    def as_dict_dataset_splits(self) -> Dict[str, "HafniaDataset"]:
        if "split" not in self.table.columns:
            raise ValueError("Dataset must contain a 'split' column.")

        splits = {}
        for split_name in SplitName.valid_splits():
            splits[split_name] = self.create_split_dataset(split_name)

        return splits

    def create_sample_dataset(self) -> "HafniaDataset":
        if "is_sample" not in self.table.columns:
            raise ValueError("Dataset must contain an 'is_sample' column.")
        table = self.table.filter(pl.col("is_sample"))
        return self.update_table(table)

    def create_split_dataset(self, split_name: Union[str | List[str]]) -> "HafniaDataset":
        if isinstance(split_name, str):
            split_names = [split_name]
        elif isinstance(split_name, list):
            split_names = split_name

        for name in split_names:
            if name not in SplitName.valid_splits():
                raise ValueError(f"Invalid split name: {split_name}. Valid splits are: {SplitName.valid_splits()}")

        filtered_dataset = self.table.filter(pl.col(ColumnName.SPLIT).is_in(split_names))
        return self.update_table(filtered_dataset)

    def get_task_by_name(self, task_name: str) -> TaskInfo:
        for task in self.info.tasks:
            if task.name == task_name:
                return task
        raise ValueError(f"Task with name {task_name} not found in dataset info.")

    def update_table(self, table: pl.DataFrame) -> "HafniaDataset":
        return HafniaDataset(info=self.info.model_copy(), table=table)

    @staticmethod
    def read_from_path(path_folder: Path, check_files_exists: bool = True) -> "HafniaDataset":
        if not path_folder.exists():
            raise FileNotFoundError(f"Path {path_folder} does not exist.")

        path_dataset_info = path_folder / FILENAME_DATASET_INFO
        if not path_dataset_info.exists():
            raise FileNotFoundError(
                f"Dataset info file '{path_dataset_info.name}' not found in '{path_dataset_info.parent}'."
            )

        dataset_info = DatasetInfo.from_json_file(path_folder / FILENAME_DATASET_INFO)
        table = read_table_from_path(path_folder)

        # Convert from relative paths to absolute paths
        table = table.with_columns(
            pl.concat_str([pl.lit(str(path_folder.absolute()) + os.sep), pl.col("file_name")]).alias("file_name")
        )
        if check_files_exists:
            check_image_paths(table)
        return HafniaDataset(table=table, info=dataset_info)

    def write(self, path_folder: Path, check_for_duplicates: bool = True) -> None:
        print(f"Writing dataset to {path_folder}...")
        if not path_folder.exists():
            path_folder.mkdir(parents=True)

        if check_for_duplicates:
            file_paths = [Path(str_path).name for str_path in self.table["file_name"].to_list()]
            is_not_unique_filenames = len(file_paths) != len(set(file_paths))
            if is_not_unique_filenames:
                raise ValueError(
                    "Dataset contains non-unique filenames - this will overwrite images when writing to disk. "
                    "The issue often occurs when the same filename is used in multiple splits or folders for different images "
                    "E.g. train/0001.jpg and val/0001.jpg or car/0001.jpg and person/0001.jpg."
                )

        new_relative_paths = []
        for org_path in tqdm(self.table["file_name"].to_list(), desc="- Copy images"):
            org_path = Path(org_path)
            if not org_path.exists():
                raise FileNotFoundError(f"File {org_path} does not exist in the dataset.")

            new_path = path_folder / "data" / Path(org_path).name
            if not new_path.exists():
                new_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(org_path, new_path)

            if not new_path.exists():
                raise FileNotFoundError(f"File {new_path} does not exist in the dataset.")
            new_relative_paths.append(str(new_path.relative_to(path_folder)))

        table = self.table.with_columns(pl.Series(new_relative_paths).alias("file_name"))
        table.write_ndjson(path_folder / FILENAME_ANNOTATIONS_JSONL)  # Json for readability
        table.write_parquet(path_folder / FILENAME_ANNOTATIONS_PARQUET)  # Parquet for speed
        self.info.write_json(path_folder / FILENAME_DATASET_INFO)

    def __eq__(self, value) -> bool:
        if not isinstance(value, HafniaDataset):
            return False

        if self.info != value.info:
            return False

        if not isinstance(self.table, pl.DataFrame) or not isinstance(value.table, pl.DataFrame):
            return False

        if not self.table.equals(value.table):
            return False
        return True

    def print_stats(self) -> None:
        t0 = time.time()
        table_base = Table(title="Dataset Statistics", show_lines=True, box=rich.box.SIMPLE)
        table_base.add_column("Property", style="cyan")
        table_base.add_column("Value")
        table_base.add_row("Dataset Name", self.info.dataset_name)
        table_base.add_row("Version", self.info.version)
        table_base.add_row("Number of samples", str(len(self.table)))
        rich.print(table_base)
        rich.print(self.info.tasks)

        splits_sets = {
            "All": SplitName.valid_splits(),
            "Train": [SplitName.TRAIN],
            "Validation": [SplitName.VAL],
            "Test": [SplitName.TEST],
        }
        rows = []
        for split_name, splits in splits_sets.items():
            dataset_split = self.create_split_dataset(splits)
            table = dataset_split.table
            row = {}
            row["Split"] = split_name
            row["Sample "] = len(table)
            for PrimitiveType in PRIMITIVE_TYPES:
                column_name = PrimitiveType.column_name()
                objects_df = create_primitive_table(table, PrimitiveType=PrimitiveType, keep_sample_data=False)
                if objects_df is None:
                    continue
                for (task_name,), object_group in objects_df.group_by(FieldName.TASK_NAME):
                    count = len(object_group[FieldName.CLASS_NAME])
                    row[f"{PrimitiveType.__name__}\n{task_name}"] = count
            rows.append(row)
        print(f"Dataset stats in {time.time() - t0:.2f} seconds.")

        rich_table = Table(title="Dataset Statistics", show_lines=True, box=rich.box.SIMPLE)
        for i_row, row in enumerate(rows):
            if i_row == 0:
                for column_name in row.keys():
                    rich_table.add_column(column_name, justify="left", style="cyan")
            rich_table.add_row(*[str(value) for value in row.values()])
        rich.print(rich_table)


def check_hafnia_dataset_from_path(path_dataset: Path) -> None:
    dataset = HafniaDataset.read_from_path(path_dataset, check_files_exists=True)
    check_hafnia_dataset(dataset)


def check_hafnia_dataset(dataset: HafniaDataset):
    print("Checking Hafnia dataset...")
    assert isinstance(dataset.info.version, str) and len(dataset.info.version) > 0
    assert isinstance(dataset.info.dataset_name, str) and len(dataset.info.dataset_name) > 0

    is_sample_list = set(dataset.table.select(pl.col(ColumnName.IS_SAMPLE)).unique().to_series().to_list())
    if True not in is_sample_list:
        raise ValueError(f"The dataset should contain '{ColumnName.IS_SAMPLE}=True' samples")

    actual_splits = dataset.table.select(pl.col(ColumnName.SPLIT)).unique().to_series().to_list()
    expected_splits = SplitName.valid_splits()
    if set(actual_splits) != set(expected_splits):
        raise ValueError(f"Expected all splits '{expected_splits}' in dataset, but got '{actual_splits}'. ")

    expected_tasks = dataset.info.tasks
    for task in expected_tasks:
        primitive = task.primitive.__name__
        column_name = task.primitive.column_name()
        primitive_column = dataset.table[column_name]
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
        if column_name not in dataset.table.columns:
            continue
        objects_df = create_primitive_table(dataset.table, PrimitiveType=PrimitiveType, keep_sample_data=False)
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


# if __name__ == "__main__":
#     from data_management.shape_primitives import COORDINATE_TYPES, Bbox, Classification
#     from PIL import Image

#     ## READ AND WRITE ###
#     ## Using a hafnia dataset
#     # path_hafnia_dataset = Path("src/data_management/encord_datasets/tiny_dataset/pipeline_data/extracted_images/hidden")
#     path_hafnia_dataset = Path(
#         "src/data_management/encord_datasets/midwest_vehicle_detection/pipeline_data/extracted_images/hidden"
#     )
#     # path_hafnia_dataset = Path("src/data_management/fiftyone_datasets/coco_2017/pipeline_data/extracted_images/hidden")

#     # Read a hafnia dataset from path
#     hafnia_dataset = HafniaDataset.read_from_path(path_hafnia_dataset)

#     polars_table: pl.DataFrame = hafnia_dataset.table  # polars table
#     dataset_info: DatasetInfo = hafnia_dataset.info  # DatasetInfo object
#     rich.print(dataset_info)

#     # stats:
#     hafnia_dataset.print_stats()

#     path_tmp = Path(".data") / "tmp_hafnia_dataset"
#     if False:
#         # Write a hafnia dataset to path
#         hafnia_dataset.write(path_tmp)

#     # Iterate dataset:
#     for sample_dict in hafnia_dataset:
#         sample = Sample(**sample_dict)
#         image = sample.read_image()  # Read image from sample
#         annotations_all = sample.get_annotations()  # Get all annotations from the sample
#         annotations_bbox = sample.get_annotations(primitive_types=[Bbox])  # Get all annotations from the sample
#         break

#     # Make visualizations
#     img_visualization = sample.draw_annotations()  # Draw annotations on the sample image
#     Image.fromarray(img_visualization).save(path_tmp / "sample_visualized.png")

#     # Dataset transforms
#     dataset_sample = hafnia_dataset.create_sample_dataset()
#     dataset_train = hafnia_dataset.create_split_dataset(SplitName.TRAIN)
#     dataset_val = hafnia_dataset.create_split_dataset(SplitName.VAL)

#     small_dataset = hafnia_dataset.sample(n_samples=10, seed=42)  # Sample 10 samples from the dataset
#     shuffled_dataset = hafnia_dataset.shuffle(seed=42)  # Shuffle the dataset

#     split_ratios = {SplitName.TRAIN: 0.8, SplitName.VAL: 0.1, SplitName.TEST: 0.1}
#     shuffled_dataset = hafnia_dataset.split_by_ratios(split_ratios)  # Shuffle the dataset

#     #### Using polars to transform and get dataset stats ###

#     # dataset: pl.DataFrame = pl.concat(dataset_spltis)
#     # pprint.pprint(dataset.schema)

#     # # Define your mapping here
#     # class_name_mapping = {
#     #     "Vehicle.Car": "car",
#     #     "Person": "person",
#     #     # Add more mappings as needed
#     # }

#     # # Remap class_name inside objects using Polars' internal functions'
#     # UNMAPPED_CLASS_NAME = "UNMAPPED_CLASS_NAME"
#     # COLUMN_NAME = "objects"
#     # FIELD_NAME = "class_name"

#     # # Rename dataset["objects"]
#     # t0 = time.time()
#     # try:
#     #     dataset = dataset.with_columns(
#     #         pl.col(COLUMN_NAME).list.eval(
#     #             pl.element().struct.with_fields(pl.field(FIELD_NAME).replace_strict(class_name_mapping).alias(FIELD_NAME))
#     #         )
#     #     )
#     # except pl.exceptions.InvalidOperationError:
#     #     raise ValueError("The provided `class_name_mapping` does not match the class names in the dataset.")

#     # dataset = dataset.with_columns(pl.col("objects").list.filter(pl.element().struct.field("class_name") == "MISSING").alias("objects"))

#     # print(f"Remapped class names in dataset using Polars' internal functions in {time.time() - t0:.2f} seconds.")
#     # dataset[0]["objects"][0][0]

#     # # Keep only car and person objects
#     # n_objects_before = dataset["objects"].list.len().sum()

#     # # Keep only samples with objects.
#     # dataset = dataset.filter(pl.col("objects").list.len() > 0)
#     # dataset = dataset.with_columns(pl.col("objects").list.filter(pl.element().struct.field("class_name").is_in(["car", "person"])))

#     # n_objects_after = dataset["objects"].list.len().sum()
#     # dataset["objects"].list.len().sum()

#     # t0 = time.time()
#     # for row in dataset.iter_rows(named=True):
#     #     pass

#     # dataset.estimated_size(unit="mb")
#     # dataset.drop(["bitmasks"]).estimated_size(unit="mb")
#     # types = dataset["Weather"].unique().to_list()
#     # dataset = dataset.cast({"Weather": pl.Enum(types)})

#     # dataset.unnest(columns=["objects"])
#     # dataset.write_ndjson()

#     ## Create a hafnia dataset
#     # Define classifications
#     classifications = [
#         Classification(
#             class_name="Sunny/Clear", class_idx=0, object_id=None, draw_label=True, task_name="Weather", meta=None
#         ),
#         Classification(
#             class_name="Twilight", class_idx=3, object_id=None, draw_label=True, task_name="Time of Day", meta=None
#         ),
#         Classification(
#             class_name="Rural",
#             class_idx=1,
#             object_id=None,
#             draw_label=True,
#             task_name="Geographical Context",
#             meta=None,
#         ),
#     ]

#     # Define objects bounding boxes. It could also be other objects Bitmask, Polygon
#     objects = [
#         Bbox(
#             height=0.10058,
#             width=0.08105,
#             top_left_x=0.54785,
#             top_left_y=0.2207,
#             class_name="Vehicle.Single_Truck",
#             class_idx=6,
#             object_id="aNJlrkmW",
#             meta=None,
#         ),
#         Bbox(
#             height=0.0322265625,
#             width=0.0361328125,
#             top_left_x=0.6318359375,
#             top_left_y=0.2666015625,
#             class_name="Vehicle.Car",
#             class_idx=3,
#             object_id="ZMeevOXB",
#             draw_label=True,
#             task_name="bboxes",
#             meta=None,
#         ),
#     ]

#     sample = Sample(
#         image_id="8",
#         file_name="data/video_968a1b31-4a22-4c3e-bf36-5cd60281954b_1fps_mp4_frame_00008.png",
#         height=1080,
#         width=1920,
#         split="validation",
#         is_sample=True,
#         frame_number=None,
#         video_name=None,
#         meta={
#             "video.data_duration": 120.0,
#             "video.data_fps": 1.0,
#             "videoDownload.from": "2024-07-10T18:30:00+0000",
#             "videoDownload.downloadStart": "2024-08-26T11:02:10+0000",
#             "dataset_hash": "9a0faaba-46d7-484a-91e8-b34b7c9ef236",
#         },
#         classifications=classifications,
#         objects=objects,
#     )

#     ## To dict:
#     sample_dict = sample.model_dump()
#     sample_json_str = sample.model_dump_json()

#     # To JSON string
#     path_file = Path("sample.json")
#     path_file.write_text(sample_json_str)

#     # And back to Sample object
#     sample_again = Sample.model_validate_json(path_file.read_text())

#     samples = []
#     for i_sample in range(10):  # Fake 10 samples
#         # Create a copy of the sample with a unique image_id
#         sample = sample.model_copy(deep=True)
#         sample.image_id = f"sample_{i_sample}"

#         samples.append(sample)

#     dataset_info = DatasetInfo(
#         dataset_name="test_dataset",
#         version="0.1.0",
#         tasks=[
#             TaskInfo(primitive=Classification, class_names=["Sunny/Clear", "Twilight", "Rural"], name="Weather"),
#             TaskInfo(primitive=Bbox, class_names=["Vehicle.Single_Truck", "Vehicle.Car"], name="bboxes"),
#         ],
#     )

#     # Create a HafniaDataset from samples
#     hafnia_dataset = HafniaDataset.from_samples(samples, info=dataset_info)
