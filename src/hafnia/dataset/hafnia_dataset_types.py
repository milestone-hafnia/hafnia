import collections
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, Union

import cv2
import more_itertools
import numpy as np
import polars as pl
from packaging.version import Version
from PIL import Image
from pydantic import BaseModel, Field, field_serializer, field_validator

import hafnia
from hafnia.dataset import dataset_helpers
from hafnia.dataset.dataset_helpers import version_from_string
from hafnia.dataset.dataset_names import (
    FILENAME_ANNOTATIONS_JSONL,
    FILENAME_ANNOTATIONS_PARQUET,
    FILENAME_DATASET_INFO,
    SampleField,
    StorageFormat,
)
from hafnia.dataset.primitives import (
    PRIMITIVE_TYPES,
    Bbox,
    Bitmask,
    Classification,
    Polygon,
    get_primitive_type_from_string,
)
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

    def get_class_index(self, class_name: str) -> int:
        """Get class index for a given class name"""
        if self.class_names is None:
            raise ValueError(f"Task '{self.name}' has no class names defined.")
        if class_name not in self.class_names:
            raise ValueError(f"Class name '{class_name}' not found in task '{self.name}'.")
        return self.class_names.index(class_name)

    # The 'primitive'-field of type 'Type[Primitive]' is not supported by pydantic out-of-the-box as
    # the 'Primitive' class is an abstract base class and for the actual primitives such as Bbox, Bitmask, Classification.
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

    def full_name(self) -> str:
        """Get qualified name for the task: <primitive_name>:<task_name>"""
        return f"{self.primitive.__name__}:{self.name}"

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
    version: str = Field(default="0.0.0", description="Version of the dataset")
    dataset_title: Optional[str] = Field(default=None, description="Optional, human-readable title of the dataset")
    description: Optional[str] = Field(default=None, description="Optional, description of the dataset")
    tasks: List[TaskInfo] = Field(default=None, description="List of tasks in the dataset")
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
        version_casted: Version = dataset_helpers.version_from_string(format_version, raise_error=True)

        if version_casted > Version(hafnia.__dataset_format_version__):
            user_logger.warning(
                f"The loaded dataset format version '{format_version}' is newer than the format version "
                f"'{hafnia.__dataset_format_version__}' used in your version of Hafnia. Please consider "
                f"updating Hafnia package."
            )
        return str(version_casted)

    @field_validator("version")
    @classmethod
    def _validate_version(cls, dataset_version: Optional[str]) -> Optional[str]:
        version_casted: Version = dataset_helpers.version_from_string(dataset_version, raise_error=True)
        return str(version_casted)

    def check_for_duplicate_task_names(self) -> List[TaskInfo]:
        return self._validate_check_for_duplicate_tasks(self.tasks)

    def write_json(self, path: Path, indent: Optional[int] = 4) -> None:
        json_str = self.model_dump_json(indent=indent)
        path.write_text(json_str)

    @staticmethod
    def from_json_file(path: Path) -> "DatasetInfo":
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
    def merge(info0: "DatasetInfo", info1: "DatasetInfo") -> "DatasetInfo":
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
        meta = (info0.meta or {}).copy()
        meta.update(info1.meta or {})
        return DatasetInfo(
            dataset_name=info0.dataset_name + "+" + info1.dataset_name,
            version="0.0.0",
            tasks=list(unique_tasks),
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

    def get_tasks_by_primitive(self, primitive: Union[Type[Primitive], str]) -> List[TaskInfo]:
        """
        Get all tasks by their primitive type.
        """
        if isinstance(primitive, str):
            primitive = get_primitive_type_from_string(primitive)

        tasks_with_primitive = [task for task in self.tasks if task.primitive == primitive]
        return tasks_with_primitive

    def get_task_by_primitive(self, primitive: Union[Type[Primitive], str]) -> TaskInfo:
        """
        Get task by its primitive type. Raises an error if the primitive type is not found or if multiple tasks
        have the same primitive type.
        """

        tasks_with_primitive = self.get_tasks_by_primitive(primitive)
        if len(tasks_with_primitive) == 0:
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
        from hafnia.dataset.operations import dataset_transformations

        task = dataset_transformations.get_task_info_from_task_name_and_primitive(
            tasks=self.tasks,
            primitive=primitive,
            task_name=task_name,
        )
        return task

    def replace_task(self, old_task: TaskInfo, new_task: Optional[TaskInfo]) -> "DatasetInfo":
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


class Sample(BaseModel):
    file_path: Optional[str] = Field(description="Path to the image/video file.")
    height: int = Field(description="Height of the image")
    width: int = Field(description="Width of the image")
    split: str = Field(description="Split name, e.g., 'train', 'val', 'test'")
    tags: List[str] = Field(
        default_factory=list,
        description="Tags for a given sample. Used for creating subsets of the dataset.",
    )
    storage_format: str = Field(
        default=StorageFormat.IMAGE,
        description="Storage format. Sample data is stored as image or inside a video or zip file.",
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
    bboxes: Optional[List[Bbox]] = Field(default=None, description="Optional list of bounding boxes")
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
        if self.file_path is None:
            raise ValueError(f"Sample has no '{SampleField.FILE_PATH}' defined.")
        path_image = Path(self.file_path)
        if not path_image.exists():
            raise FileNotFoundError(f"Image file {path_image} does not exist. Please check the file path.")

        image = Image.open(str(path_image))
        return image

    def read_image(self) -> np.ndarray:
        if self.storage_format == StorageFormat.VIDEO:
            video = cv2.VideoCapture(str(self.file_path))
            if self.collection_index is None:
                raise ValueError("collection_index must be set for video storage format to read the correct frame.")
            video.set(cv2.CAP_PROP_POS_FRAMES, self.collection_index)
            success, image = video.read()
            video.release()
            if not success:
                raise ValueError(f"Could not read frame {self.collection_index} from video file {self.file_path}.")
            return image

        elif self.storage_format == StorageFormat.IMAGE:
            image_pil = self.read_image_pillow()
            image = np.array(image_pil)
        else:
            raise ValueError(f"Unsupported storage format: {self.storage_format}")
        return image

    def draw_annotations(self, image: Optional[np.ndarray] = None) -> np.ndarray:
        from hafnia.visualizations import image_visualizations

        if image is None:
            image = self.read_image()
        annotations = self.get_annotations()
        annotations_visualized = image_visualizations.draw_annotations(image=image, primitives=annotations)
        return annotations_visualized


@dataclass
class DatasetMetadataFilePaths:
    dataset_info: str  # Use 'str' to also support s3 paths
    annotations_jsonl: Optional[str]
    annotations_parquet: Optional[str]

    def as_list(self) -> List[str]:
        files = [self.dataset_info]
        if self.annotations_jsonl is not None:
            files.append(self.annotations_jsonl)
        if self.annotations_parquet is not None:
            files.append(self.annotations_parquet)
        return files

    def read_samples(self) -> pl.DataFrame:
        if self.annotations_parquet is not None:
            if not Path(self.annotations_parquet).exists():
                raise FileNotFoundError(f"Parquet annotations file '{self.annotations_parquet}' does not exist.")
            user_logger.info(f"Reading dataset annotations from Parquet file: {self.annotations_parquet}")
            return pl.read_parquet(self.annotations_parquet)

        if self.annotations_jsonl is not None:
            if not Path(self.annotations_jsonl).exists():
                raise FileNotFoundError(f"JSONL annotations file '{self.annotations_jsonl}' does not exist.")
            user_logger.info(f"Reading dataset annotations from JSONL file: {self.annotations_jsonl}")
            return pl.read_ndjson(self.annotations_jsonl)

        raise ValueError(
            "No annotations file available to read samples from. Dataset is missing both JSONL and Parquet files."
        )

    @staticmethod
    def from_path(path_dataset: Path) -> "DatasetMetadataFilePaths":
        path_dataset = path_dataset.absolute()
        metadata_files = DatasetMetadataFilePaths(
            dataset_info=str(path_dataset / FILENAME_DATASET_INFO),
            annotations_jsonl=str(path_dataset / FILENAME_ANNOTATIONS_JSONL),
            annotations_parquet=str(path_dataset / FILENAME_ANNOTATIONS_PARQUET),
        )

        return metadata_files

    @staticmethod
    def available_versions_from_files_list(files: list[str]) -> Dict[Version, "DatasetMetadataFilePaths"]:
        versions_and_files: Dict[Version, Dict[str, str]] = collections.defaultdict(dict)
        for metadata_file in files:
            version_str, filename = metadata_file.split("/")[-2:]
            versions_and_files[version_str][filename] = metadata_file

        available_versions: Dict[Version, DatasetMetadataFilePaths] = {}
        for version_str, version_files in versions_and_files.items():
            version_casted: Version = dataset_helpers.version_from_string(version_str, raise_error=False)
            if version_casted is None:
                continue

            if FILENAME_DATASET_INFO not in version_files:
                continue
            dataset_metadata_file = DatasetMetadataFilePaths(
                dataset_info=version_files[FILENAME_DATASET_INFO],
                annotations_jsonl=version_files.get(FILENAME_ANNOTATIONS_JSONL, None),
                annotations_parquet=version_files.get(FILENAME_ANNOTATIONS_PARQUET, None),
            )

            available_versions[version_casted] = dataset_metadata_file

        return available_versions

    def check_version(self, version: str, raise_error: bool = True) -> bool:
        """
        Check if the dataset metadata files match the given version.
        If raise_error is True, raises ValueError if the version does not match.
        """
        valid_version = version_from_string(version, raise_error=raise_error)
        if valid_version is None:
            return False

        path_dataset_info = Path(self.dataset_info)
        if not path_dataset_info.exists():
            raise FileNotFoundError(f"Dataset info file missing '{self.dataset_info}' in dataset folder.")

        dataset_info = json.loads(path_dataset_info.read_text())
        dataset_version = dataset_info.get("version", None)
        if dataset_version != version:
            if raise_error:
                raise ValueError(
                    f"Dataset version mismatch. Expected version '{version}' but found "
                    f"version '{dataset_version}' in dataset info."
                )
            return False

        return True

    def exists(self, version: Optional[str] = None, raise_error: bool = True) -> bool:
        """
        Check if all metadata files exist.
        Add version to check if it matches the version in dataset info.
        If raise_error is True, raises FileNotFoundError if any file is missing.
        """
        path_dataset_info = Path(self.dataset_info)
        if not path_dataset_info.exists():
            if raise_error:
                raise FileNotFoundError(f"Dataset info file missing '{self.dataset_info}' in dataset folder.")
            return False

        if version is not None and self.check_version(version, raise_error=raise_error) is False:
            return False

        has_jsonl_file = self.annotations_jsonl is not None and Path(self.annotations_jsonl).exists()
        if has_jsonl_file:
            return True

        has_parquet_file = self.annotations_parquet is not None and Path(self.annotations_parquet).exists()
        if has_parquet_file:
            return True

        if raise_error:
            raise FileNotFoundError(
                f"Missing annotation file. Expected either '{FILENAME_ANNOTATIONS_JSONL}' or "
                f"'{FILENAME_ANNOTATIONS_PARQUET}' in dataset folder."
            )

        return False
