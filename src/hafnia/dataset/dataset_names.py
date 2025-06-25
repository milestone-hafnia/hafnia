from enum import Enum
from typing import List

FILENAME_DATASET_INFO = "dataset_info.json"
FILENAME_ANNOTATIONS_JSONL = "annotations.jsonl"
FILENAME_ANNOTATIONS_PARQUET = "annotations.parquet"

DATASET_FILENAMES = [
    FILENAME_DATASET_INFO,
    FILENAME_ANNOTATIONS_JSONL,
    FILENAME_ANNOTATIONS_PARQUET,
]


class DeploymentStage(Enum):
    STAGING = "staging"
    PRODUCTION = "production"


class FieldName:
    CLASS_NAME: str = "class_name"  # Name of the class this primitive is associated with, e.g. "car" for Bbox
    CLASS_IDX: str = (
        "class_idx"  # Index of the class this primitive is associated with, e.g. 0 for "car" if it is the first class
    )
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
    FILE_NAME: str = "file_name"
    HEIGHT: str = "height"
    WIDTH: str = "width"
    SPLIT: str = "split"
    IS_SAMPLE: str = "is_sample"
    REMOTE_PATH: str = "remote_path"  # Path to the file in remote storage, e.g. S3
    META: str = "meta"


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
