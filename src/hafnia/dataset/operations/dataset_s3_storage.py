import tempfile
import time
from pathlib import Path
from typing import Dict, Optional

import polars as pl

from hafnia.dataset.dataset_helpers import hash_file_xxhash
from hafnia.dataset.dataset_names import (
    DatasetVariant,
    ResourceCredentials,
    SampleField,
)
from hafnia.dataset.hafnia_dataset import HafniaDataset
from hafnia.log import user_logger
from hafnia.platform import s5cmd_utils
from hafnia.platform.datasets import get_upload_credentials
from hafnia.utils import progress_bar
from hafnia_cli.config import Config


def delete_hafnia_dataset_files_on_platform(
    dataset_name: str,
    interactive: bool = True,
    cfg: Optional[Config] = None,
) -> bool:
    cfg = cfg or Config()
    resource_credentials = get_upload_credentials(dataset_name, cfg=cfg)

    if resource_credentials is None:
        raise RuntimeError("Failed to get upload credentials from the platform.")

    return delete_hafnia_dataset_files_from_resource_credentials(
        interactive=interactive,
        resource_credentials=resource_credentials,
    )


def delete_hafnia_dataset_files_from_resource_credentials(
    resource_credentials: ResourceCredentials,
    interactive: bool = True,
) -> bool:
    envs = resource_credentials.aws_credentials()
    bucket_name = resource_credentials.bucket_name()
    if interactive:
        confirmation = (
            input(
                f"WARNING THIS WILL delete all files stored in 's3://{bucket_name}'.\n"
                "Meaning that all previous versions of the dataset will be deleted. \n"
                "Normally this is not needed, but if you have changed the dataset structure or want to start from fresh, "
                "you can delete all files in the S3 bucket. "
                "\nDo you really want to delete all files? (yes/NO): "
            )
            .strip()
            .lower()
        )
        if confirmation != "yes":
            user_logger.info("Delete operation cancelled by the user.")
            return False
    user_logger.info(f"Deleting all files in S3 bucket '{bucket_name}'...")
    s5cmd_utils.delete_bucket_content(bucket_prefix=f"s3://{bucket_name}", remove_bucket=True, append_envs=envs)
    return True


def sync_hafnia_dataset_to_s3(
    dataset: HafniaDataset,
    bucket_prefix: str,
    allow_version_overwrite: bool = False,
    interactive: bool = True,
    envs: Optional[Dict[str, str]] = None,
) -> None:
    t0 = time.time()
    # bucket_prefix e.g. 's3://bucket-name/sample'
    remote_paths = []
    for file_str in progress_bar(dataset.samples[SampleField.FILE_PATH], description="Hashing data files"):
        path_file = Path(file_str)
        file_hash = hash_file_xxhash(path_file)

        # Relative path in S3 bucket e.g. 'data/e2/b0/e2b000ac47b19a999bee5456a6addb88.png'
        relative_path = s3_prefix_from_hash(hash=file_hash, suffix=path_file.suffix)

        # Remote path in S3 bucket e.g. 's3://bucket-name/sample/data/e2/b0/e2b000ac47b19a999bee5456a6addb88.png'
        remote_path = f"{bucket_prefix}/{relative_path}"
        remote_paths.append(remote_path)

    dataset.samples = dataset.samples.with_columns(pl.Series(remote_paths).alias(SampleField.REMOTE_PATH))

    user_logger.info(f"Syncing dataset to S3 bucket '{bucket_prefix}'")
    files_in_s3 = set(s5cmd_utils.list_bucket(bucket_prefix=bucket_prefix, append_envs=envs))

    # Discover data files (images, videos, etc.) missing in s3
    data_files_missing = dataset.samples.filter(~pl.col(SampleField.REMOTE_PATH).is_in(files_in_s3))
    files_already_in_s3 = dataset.samples.filter(pl.col(SampleField.REMOTE_PATH).is_in(files_in_s3))

    with tempfile.TemporaryDirectory() as temp_dir:  # Temp folder to store metadata files
        path_temp = Path(temp_dir)
        # File paths are dropped when uploading to S3
        dataset = dataset.update_samples(dataset.samples.drop(SampleField.FILE_PATH))
        dataset.write_annotations(path_temp)

        # Discover versioned metadata files (e.g. "annotations.jsonl", "dataset_info.json") missing in s3
        metadata_files_local = []
        metadata_files_s3 = []
        for filename in path_temp.iterdir():
            metadata_files_s3.append(f"{bucket_prefix}/versions/{dataset.info.version}/{filename.name}")
            metadata_files_local.append(filename.as_posix())

        overwrite_metadata_files = files_in_s3.intersection(set(metadata_files_s3))
        will_overwrite_metadata_files = len(overwrite_metadata_files) > 0

        n_files_already_in_s3 = len(files_already_in_s3)
        user_logger.info(f"Sync dataset to {bucket_prefix}")
        user_logger.info(
            f"- Found that {n_files_already_in_s3} / {len(dataset.samples)} data files already exist. "
            f"Meaning {len(data_files_missing)} data files will be uploaded. \n"
            f"- Will upload {len(metadata_files_local)} metadata files. \n"
            f"- Total files to upload: {len(data_files_missing) + len(metadata_files_local)}"
        )
        if will_overwrite_metadata_files:
            msg = f"Metadata files for dataset version '{dataset.info.version}' already exist"
            if allow_version_overwrite:
                user_logger.warning(
                    f"- WARNING: {msg}. Version will be overwritten as 'allow_version_overwrite=True' is set."
                )
            else:
                raise ValueError(
                    f"Upload cancelled. {msg}. \nTo overwrite existing metadata files, "
                    "you will need to set 'allow_version_overwrite=True' explicitly."
                )

        has_missing_files = len(data_files_missing) > 0
        if interactive and (has_missing_files or will_overwrite_metadata_files):
            print("Please type 'yes' to upload files.")
            confirmation = input("Do you want to continue? (yes/NO): ").strip().lower()

            if confirmation != "yes":
                raise RuntimeError("Upload cancelled by user.")

        local_paths = metadata_files_local + data_files_missing[SampleField.FILE_PATH].to_list()
        s3_paths = metadata_files_s3 + data_files_missing[SampleField.REMOTE_PATH].to_list()
        s5cmd_utils.fast_copy_files(local_paths, s3_paths, append_envs=envs, description="Uploading files")
    user_logger.info(f"- Synced dataset in {time.time() - t0:.2f} seconds.")


def sync_dataset_files_to_platform(
    dataset: HafniaDataset,
    sample_dataset: Optional[HafniaDataset] = None,
    interactive: bool = True,
    allow_version_overwrite: bool = False,
    cfg: Optional[Config] = None,
) -> None:
    cfg = cfg or Config()
    resource_credentials = get_upload_credentials(dataset.info.dataset_name, cfg=cfg)

    if resource_credentials is None:
        raise RuntimeError("Failed to get upload credentials from the platform.")

    sync_dataset_files_to_platform_from_resource_credentials(
        dataset=dataset,
        sample_dataset=sample_dataset,
        interactive=interactive,
        allow_version_overwrite=allow_version_overwrite,
        resource_credentials=resource_credentials,
    )


def sync_dataset_files_to_platform_from_resource_credentials(
    dataset: HafniaDataset,
    sample_dataset: Optional[HafniaDataset],
    interactive: bool,
    allow_version_overwrite: bool,
    resource_credentials: ResourceCredentials,
):
    envs = resource_credentials.aws_credentials()
    bucket_name = resource_credentials.bucket_name()

    for dataset_variant_type in [DatasetVariant.SAMPLE, DatasetVariant.HIDDEN]:
        if dataset_variant_type == DatasetVariant.SAMPLE:
            if sample_dataset is None:
                dataset_variant = dataset.create_sample_dataset()
            else:
                dataset_variant = sample_dataset
        else:
            dataset_variant = dataset

        sync_hafnia_dataset_to_s3(
            dataset=dataset_variant,
            bucket_prefix=f"s3://{bucket_name}/{dataset_variant_type.value}",
            interactive=interactive,
            allow_version_overwrite=allow_version_overwrite,
            envs=envs,
        )


def s3_prefix_from_hash(hash: str, suffix: str) -> str:
    """
    Generate a relative S3 path from a hash value for objects stored in S3.

    This function deliberately uses a hierarchical directory layout based on the
    hash prefix to avoid putting too many objects in a single S3 prefix, which
    can run into AWS S3 rate limits and performance issues. For example, for
    hash "dfe8f3b1c2a4f5b6c7d8e9f0a1b2c3d4" and suffix ".png", the returned
    path will be:

        "data/df/e8/dfe8f3b1c2a4f5b6c7d8e9f0a1b2c3d4.png"

    Note: This intentionally differs from when images are stored to disk locally, where
    a flat path of the form ``data/<hash><suffix>`` is used.
    """
    s3_prefix = f"data/{hash[:2]}/{hash[2:4]}/{hash}{suffix}"
    return s3_prefix
