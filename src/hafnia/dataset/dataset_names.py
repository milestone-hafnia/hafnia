from enum import Enum
from typing import List

FILENAME_RECIPE_JSON = "recipe.json"
FILENAME_DATASET_INFO = "dataset_info.json"
FILENAME_ANNOTATIONS_JSONL = "annotations.jsonl"
FILENAME_ANNOTATIONS_PARQUET = "annotations.parquet"

DATASET_FILENAMES_REQUIRED = [
    FILENAME_DATASET_INFO,
    FILENAME_ANNOTATIONS_JSONL,
    FILENAME_ANNOTATIONS_PARQUET,
]


class DeploymentStage(Enum):
    STAGING = "staging"
    PRODUCTION = "production"


TAG_IS_SAMPLE = "sample"

OPS_REMOVE_CLASS = "__REMOVE__"


class FieldName:
    CLASS_NAME: str = "class_name"  # Name of the class this primitive is associated with, e.g. "car" for Bbox
    CLASS_IDX: str = "class_idx"  # Index of the class this primitive is associated with, e.g. 0 for "car" if it is the first class  # noqa: E501
    OBJECT_ID: str = "object_id"  # Unique identifier for the object, e.g. "12345123"
    CONFIDENCE: str = "confidence"  # Confidence score (0-1.0) for the primitive, e.g. 0.95 for Bbox

    META: str = "meta"  # Contains metadata about each primitive, e.g. attributes color, occluded, iscrowd, etc.
    TASK_NAME: str = "task_name"  # Name of the task this primitive is associated with, e.g. "bboxes" for Bbox

    @staticmethod
    def fields() -> List[str]:
        """
        Returns a list of expected field names for primitives.
        """
        return [
            FieldName.CLASS_NAME,
            FieldName.CLASS_IDX,
            FieldName.OBJECT_ID,
            FieldName.CONFIDENCE,
            FieldName.META,
            FieldName.TASK_NAME,
        ]


class ColumnName:
    SAMPLE_INDEX: str = "sample_index"
    FILE_PATH: str = "file_path"
    HEIGHT: str = "height"
    WIDTH: str = "width"
    SPLIT: str = "split"
    REMOTE_PATH: str = "remote_path"  # Path to the file in remote storage, e.g. S3
    ATTRIBUTION: str = "attribution"  # Attribution for the sample (image/video), e.g. creator, license, source, etc.
    TAGS: str = "tags"
    META: str = "meta"
    DATASET_NAME: str = "dataset_name"


class SplitName:
    TRAIN = "train"
    VAL = "validation"
    TEST = "test"
    UNDEFINED = "UNDEFINED"

    @staticmethod
    def valid_splits() -> List[str]:
        return [SplitName.TRAIN, SplitName.VAL, SplitName.TEST]


class DatasetVariant(Enum):
    DUMP = "dump"
    SAMPLE = "sample"
    HIDDEN = "hidden"
