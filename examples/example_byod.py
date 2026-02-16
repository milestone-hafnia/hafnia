import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

import boto3
from botocore.exceptions import ClientError

from hafnia.dataset.dataset_details_uploader import (
    upload_dataset_details_to_platform,
)
from hafnia.dataset.dataset_names import (
    ResourceCredentials,
    SampleField,
)
from hafnia.dataset.hafnia_dataset import HafniaDataset
from hafnia.dataset.hafnia_dataset_types import DatasetInfo
from hafnia.dataset.operations.dataset_s3_storage import delete_hafnia_dataset_files, sync_hafnia_dataset_to_platform
from hafnia.platform.datasets import get_dataset_by_name, get_or_create_dataset

IS_OLD_BACKEND = True


class FakeSettings:
    EXPERIMENT_DATA_READ_ROLE_ARN = "arn:aws:iam::203918851348:role/s2m-staging-dipdatalib-d819b0c1"
    DATASET_WRITE_ROLE_ARN = "arn:aws:iam::203918851348:role/dataset_writer-staging-dipdatalib-d819b0c1"


class FakeUser:
    id: int = 1234


settings = FakeSettings()


def get_bucket_region(bucket_name: str) -> str:
    s3_client = boto3.client("s3")
    response = s3_client.get_bucket_location(Bucket=bucket_name)
    region = response["LocationConstraint"]
    if region is None:
        region = "us-east-1"  # Buckets in us-east-1 return None for location
    if region == "EU":
        region = "eu-west-1"  # Handle legacy EU region

    return region


def create_s3_client_with_credentials(credentials: Optional[Dict] = None):
    if credentials:
        s3_client = boto3.client(
            "s3",
            aws_access_key_id=credentials.get("AccessKeyId"),
            aws_secret_access_key=credentials.get("SecretAccessKey"),
            aws_session_token=credentials.get("SessionToken"),
        )
    else:
        s3_client = boto3.client("s3")
    return s3_client


def create_bucket_if_missing(bucket_name: str, credentials: Optional[Dict] = None):
    if not bucket_exists(bucket_name, credentials):
        s3_client = create_s3_client_with_credentials(credentials)
        region = s3_client.meta.region_name  # Get the region from the client
        if region == "us-east-1":
            create_bucket_config = {}
        else:
            create_bucket_config = {"LocationConstraint": region}
        s3_client.create_bucket(Bucket=bucket_name, CreateBucketConfiguration=create_bucket_config)


def bucket_exists(bucket_name: str, credentials: Optional[Dict] = None) -> bool:
    s3_client = create_s3_client_with_credentials(credentials)
    try:
        s3_client.head_bucket(Bucket=bucket_name)
        return True
    except ClientError:
        return False


def get_read_dataset_credentials(bucket_name: str, user, duration: int = 3600, variant_type="sample"):
    sts_client = boto3.client("sts")
    role_arn = settings.EXPERIMENT_DATA_READ_ROLE_ARN
    role_session_name = f"DjangoS3DatasetAccess-{user.id}"

    arn_bucket = f"arn:aws:s3:::{bucket_name}"
    variant_s3_path = f"{arn_bucket}/{variant_type}"

    # Assume a role to get temporary credentials
    assumed_role_object = sts_client.assume_role(
        RoleArn=role_arn,
        RoleSessionName=role_session_name,
        Policy=json.dumps(
            {
                "Version": "2012-10-17",
                "Statement": [
                    {
                        "Sid": "AllowGetObject",
                        "Effect": "Allow",
                        "Action": ["s3:GetObject"],
                        "Resource": [f"{variant_s3_path}/*"],
                    },
                    {
                        "Sid": "AllowListPrefix",
                        "Effect": "Allow",
                        "Action": ["s3:ListBucket"],
                        "Resource": [arn_bucket],
                        "Condition": {"StringLike": {"s3:prefix": [f"{variant_type}/*", variant_type]}},
                    },
                ],
            }
        ),
        DurationSeconds=duration,  # Permission is given for 'duration' seconds
    )

    credentials = assumed_role_object["Credentials"]

    region = get_bucket_region(bucket_name=bucket_name)

    return {
        "s3_path": variant_s3_path,
        "dataset_variant": variant_type,
        "access_key": credentials["AccessKeyId"],
        "secret_key": credentials["SecretAccessKey"],
        "session_token": credentials["SessionToken"],
        "region": region,
    }


def get_write_dataset_credentials(bucket_name: str, user: FakeUser, duration: int = 3600):
    sts_client = boto3.client("sts")

    role_arn = settings.DATASET_WRITE_ROLE_ARN
    role_session_name = f"DjangoS3DatasetWriteAccess-{user.id}"
    arn_bucket = f"arn:aws:s3:::{bucket_name}"

    # Assume a role to get temporary credentials with write permissions
    assumed_role_object = sts_client.assume_role(
        RoleArn=role_arn,
        RoleSessionName=role_session_name,
        Policy=json.dumps(
            {
                "Version": "2012-10-17",
                "Statement": [
                    {
                        "Sid": "AllowBucketOperations",
                        "Effect": "Allow",
                        "Action": ["s3:CreateBucket", "s3:DeleteBucket"],
                        "Resource": [arn_bucket],
                    },
                    {
                        "Sid": "AllowObjectOperations",
                        "Effect": "Allow",
                        "Action": ["s3:PutObject", "s3:DeleteObject", "s3:PutObject"],
                        "Resource": [f"{arn_bucket}/*"],
                    },
                    {
                        "Sid": "AllowListBucket",
                        "Effect": "Allow",
                        "Action": ["s3:ListBucket"],
                        "Resource": [arn_bucket],
                    },
                ],
            }
        ),
        DurationSeconds=duration,
    )

    credentials = assumed_role_object["Credentials"]
    create_bucket_if_missing(bucket_name=bucket_name, credentials=credentials)
    region = get_bucket_region(bucket_name=bucket_name)

    return {
        "s3_path": arn_bucket,
        "access_key": credentials["AccessKeyId"],
        "secret_key": credentials["SecretAccessKey"],
        "session_token": credentials["SessionToken"],
        "region": region,
    }


############################
# Create dataset in platform
def get_or_create_dataset_on_platform(dataset: HafniaDataset) -> Dict[str, Any]:
    return get_or_create_dataset(dataset.info.dataset_name)


def get_bucket_write_credentials(dataset: HafniaDataset) -> ResourceCredentials:
    if IS_OLD_BACKEND:
        response = get_dataset_by_name(dataset.info.dataset_name)
        short_dataset_id = response["id"][-6:]
        bucket_name = f"dataset-writeable-{dataset.info.dataset_name}-{short_dataset_id}"
        credentials_dict = get_write_dataset_credentials(bucket_name=bucket_name, user=FakeUser())
        return ResourceCredentials.fix_naming(credentials_dict)

    raise NotImplementedError("New backend not implemented in this example.")


def get_bucket_read_credentials(dataset: HafniaDataset, variant_type: str = "sample") -> ResourceCredentials:
    if IS_OLD_BACKEND:
        response = get_dataset_by_name(dataset.info.dataset_name)
        short_dataset_id = response["id"][-6:]
        bucket_name = f"dataset-writeable-{dataset.info.dataset_name}-{short_dataset_id}"
        credentials_dict = get_read_dataset_credentials(
            bucket_name=bucket_name,
            user=FakeUser(),
            variant_type=variant_type,
        )
        return ResourceCredentials.fix_naming(credentials_dict)

    raise NotImplementedError("New backend not implemented in this example.")


if __name__ == "__main__":
    os.environ["AWS_PROFILE"] = "control-staging"
    user_dataset = HafniaDataset.from_name("mnist", force_redownload=True)  # 200 samples

    # Create hafnia dataset
    # user_dataset = HafniaDataset.from_name_public_dataset("mnist")  # 70000 samples

    # user_dataset = HafniaDataset.from_path(Path(".data/mnist_full"))
    # user_dataset = user_dataset.define_sample_set_by_size(n_samples=200, seed=42)
    # user_dataset = user_dataset.split_into_multiple_splits(
    #    split_name=SplitName.TEST, split_ratios={SplitName.TEST: 0.5, SplitName.VAL: 0.5}
    # )

    user_dataset.print_stats()
    dataset_name = "first-byod-dataset"

    # Update dataset info
    user_dataset.info = DatasetInfo(
        dataset_name=dataset_name,
        title="My First BYOD Dataset",
        description="This is my first Bring Your Own Data (BYOD) dataset uploaded using Hafnia.",
        tasks=user_dataset.info.tasks,
        version="0.0.1",
    )
    resource_credentials = get_bucket_write_credentials(user_dataset)
    # Create and upload dataset details
    response = get_or_create_dataset_on_platform(user_dataset)
    if True:
        # Select gallery images by name
        n_gallery_images = 9
        gallery_images = user_dataset.samples[SampleField.FILE_PATH].str.split("/").list.last().head(n_gallery_images)
        upload_dataset_details_to_platform(
            dataset=user_dataset,
            gallery_image_names=gallery_images,
            path_gallery_images=Path(f".data/gallery_images/{dataset_name}"),
            upload_gallery_images=True,
        )
        interactive = False
        delete_bucket = False
        allow_version_overwrite = True
        if delete_bucket:
            delete_hafnia_dataset_files(
                dataset=user_dataset,
                interactive=interactive,
                resource_credentials=resource_credentials,
            )

        sync_hafnia_dataset_to_platform(
            dataset=user_dataset,
            interactive=interactive,
            resource_credentials=resource_credentials,
            allow_version_overwrite=allow_version_overwrite,
        )

    selected_version = "0.0.1"
    credentials = get_bucket_read_credentials(user_dataset, variant_type="sample")

    print("Done!")
