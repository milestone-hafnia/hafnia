from enum import Enum
from typing import Dict, List, Optional

import boto3
from pydantic import BaseModel, field_validator

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


class PrimitiveField:
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
            PrimitiveField.CLASS_NAME,
            PrimitiveField.CLASS_IDX,
            PrimitiveField.OBJECT_ID,
            PrimitiveField.CONFIDENCE,
            PrimitiveField.META,
            PrimitiveField.TASK_NAME,
        ]


class SampleField:
    FILE_PATH: str = "file_path"
    HEIGHT: str = "height"
    WIDTH: str = "width"
    SPLIT: str = "split"
    TAGS: str = "tags"

    CLASSIFICATIONS: str = "classifications"
    BBOXES: str = "bboxes"
    BITMASKS: str = "bitmasks"
    POLYGONS: str = "polygons"

    STORAGE_FORMAT: str = "storage_format"  # E.g. "image", "video", "zip"
    COLLECTION_INDEX: str = "collection_index"
    COLLECTION_ID: str = "collection_id"
    REMOTE_PATH: str = "remote_path"  # Path to the file in remote storage, e.g. S3
    SAMPLE_INDEX: str = "sample_index"

    ATTRIBUTION: str = "attribution"  # Attribution for the sample (image/video), e.g. creator, license, source, etc.
    META: str = "meta"
    DATASET_NAME: str = "dataset_name"


class StorageFormat:
    IMAGE: str = "image"
    VIDEO: str = "video"
    ZIP: str = "zip"


class SplitName:
    TRAIN: str = "train"
    VAL: str = "validation"
    TEST: str = "test"
    UNDEFINED: str = "UNDEFINED"

    @staticmethod
    def valid_splits() -> List[str]:
        return [SplitName.TRAIN, SplitName.VAL, SplitName.TEST]

    @staticmethod
    def all_split_names() -> List[str]:
        return [*SplitName.valid_splits(), SplitName.UNDEFINED]


class DatasetVariant(Enum):
    DUMP = "dump"
    SAMPLE = "sample"
    HIDDEN = "hidden"


class AwsCredentials(BaseModel):
    access_key: str
    secret_key: str
    session_token: str
    region: Optional[str]

    def aws_credentials(self) -> Dict[str, str]:
        """
        Returns the AWS credentials as a dictionary.
        """
        environment_vars = {
            "AWS_ACCESS_KEY_ID": self.access_key,
            "AWS_SECRET_ACCESS_KEY": self.secret_key,
            "AWS_SESSION_TOKEN": self.session_token,
        }
        if self.region:
            environment_vars["AWS_REGION"] = self.region

        return environment_vars

    @staticmethod
    def from_session(session: boto3.Session) -> "AwsCredentials":
        """
        Creates AwsCredentials from a Boto3 session.
        """
        frozen_credentials = session.get_credentials().get_frozen_credentials()
        return AwsCredentials(
            access_key=frozen_credentials.access_key,
            secret_key=frozen_credentials.secret_key,
            session_token=frozen_credentials.token,
            region=session.region_name,
        )


ARN_PREFIX = "arn:aws:s3:::"


class ResourceCredentials(AwsCredentials):
    s3_arn: str

    @staticmethod
    def fix_naming(payload: Dict[str, str]) -> "ResourceCredentials":
        """
        The endpoint returns a payload with a key called 's3_path', but it
        is actually an ARN path (starts with arn:aws:s3::). This method renames it to 's3_arn' for consistency.
        """
        if "s3_path" in payload and payload["s3_path"].startswith(ARN_PREFIX):
            payload["s3_arn"] = payload.pop("s3_path")

        if "region" not in payload:
            payload["region"] = "eu-west-1"
        return ResourceCredentials(**payload)

    @field_validator("s3_arn")
    @classmethod
    def validate_s3_arn(cls, value: str) -> str:
        """Validate s3_arn to ensure it starts with 'arn:aws:s3:::'"""
        if not value.startswith("arn:aws:s3:::"):
            raise ValueError(f"Invalid S3 ARN: {value}. It should start with 'arn:aws:s3:::'")
        return value

    def s3_path(self) -> str:
        """
        Extracts the S3 path from the ARN.
        Example: arn:aws:s3:::my-bucket/my-prefix -> my-bucket/my-prefix
        """
        return self.s3_arn[len(ARN_PREFIX) :]

    def s3_uri(self) -> str:
        """
        Converts the S3 ARN to a URI format.
        Example: arn:aws:s3:::my-bucket/my-prefix -> s3://my-bucket/my-prefix
        """
        return f"s3://{self.s3_path()}"

    def bucket_name(self) -> str:
        """
        Extracts the bucket name from the S3 ARN.
        Example: arn:aws:s3:::my-bucket/my-prefix -> my-bucket
        """
        return self.s3_path().split("/")[0]

    def object_key(self) -> str:
        """
        Extracts the object key from the S3 ARN.
        Example: arn:aws:s3:::my-bucket/my-prefix -> my-prefix
        """
        return "/".join(self.s3_path().split("/")[1:])
