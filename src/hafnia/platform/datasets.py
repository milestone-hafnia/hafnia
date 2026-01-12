import collections
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

import rich
from packaging.version import Version
from rich import print as rprint

from hafnia import http, utils
from hafnia.dataset.dataset_names import DATASET_FILENAMES_REQUIRED, ResourceCredentials
from hafnia.dataset.dataset_recipe.dataset_recipe import (
    DatasetRecipe,
    get_dataset_path_from_recipe,
)
from hafnia.dataset.hafnia_dataset import HafniaDataset
from hafnia.http import fetch, post
from hafnia.log import user_logger
from hafnia.platform import s5cmd_utils
from hafnia.platform.download import get_resource_credentials
from hafnia.utils import timed
from hafnia_cli.config import Config


@timed("Fetching dataset by name.")
def get_dataset_by_name(dataset_name: str, cfg: Optional[Config] = None) -> Optional[Dict[str, Any]]:
    """Get dataset details by name from the Hafnia platform."""
    cfg = cfg or Config()
    endpoint_dataset = cfg.get_platform_endpoint("datasets")
    header = {"Authorization": cfg.api_key}
    full_url = f"{endpoint_dataset}?name__iexact={dataset_name}"
    datasets: List[Dict[str, Any]] = http.fetch(full_url, headers=header)  # type: ignore[assignment]
    if len(datasets) == 0:
        return None

    if len(datasets) > 1:
        raise ValueError(f"Multiple datasets found with the name '{dataset_name}'.")

    return datasets[0]


@timed("Fetching dataset by ID.")
def get_dataset_by_id(dataset_id: str, cfg: Optional[Config] = None) -> Optional[Dict[str, Any]]:
    """Get dataset details by ID from the Hafnia platform."""
    cfg = cfg or Config()
    endpoint_dataset = cfg.get_platform_endpoint("datasets")
    header = {"Authorization": cfg.api_key}
    full_url = f"{endpoint_dataset}/{dataset_id}"
    dataset: Dict[str, Any] = http.fetch(full_url, headers=header)  # type: ignore[assignment]
    if not dataset:
        return None

    return dataset


def get_or_create_dataset(dataset_name: str = "", cfg: Optional[Config] = None) -> Dict[str, Any]:
    """Create a new dataset on the Hafnia platform."""
    cfg = cfg or Config()
    dataset = get_dataset_by_name(dataset_name, cfg)

    if dataset is not None:
        user_logger.info(f"Dataset '{dataset_name}' already exists on the Hafnia platform.")
        return dataset

    endpoint_dataset = cfg.get_platform_endpoint("datasets")
    header = {"Authorization": cfg.api_key}
    dataset_title = dataset_name.replace("-", " ").title()  # convert dataset-name to title "Dataset Name"
    payload = {
        "title": dataset_title,
        "name": dataset_name,
        "overview": "No description provided.",
    }

    dataset = http.post(endpoint_dataset, headers=header, data=payload)  # type: ignore[assignment]

    # TODO: Handle issue when dataset creation fails because name is taken by another user from a different organization
    if not dataset:
        raise ValueError("Failed to create dataset on the Hafnia platform. ")

    return dataset


@timed("Fetching dataset list.")
def get_datasets(cfg: Optional[Config] = None) -> List[Dict[str, str]]:
    """List available datasets on the Hafnia platform."""
    cfg = cfg or Config()
    endpoint_dataset = cfg.get_platform_endpoint("datasets")
    header = {"Authorization": cfg.api_key}
    datasets: List[Dict[str, str]] = fetch(endpoint_dataset, headers=header)  # type: ignore
    if not datasets:
        raise ValueError("No datasets found on the Hafnia platform.")

    return datasets


@timed("Fetching dataset info.")
def get_dataset_id(dataset_name: str, endpoint: str, api_key: str) -> str:
    headers = {"Authorization": api_key}
    full_url = f"{endpoint}?name__iexact={dataset_name}"
    dataset_responses: List[Dict] = http.fetch(full_url, headers=headers)  # type: ignore[assignment]
    if not dataset_responses:
        raise ValueError(f"Dataset '{dataset_name}' was not found in the dataset library.")
    try:
        return dataset_responses[0]["id"]
    except (IndexError, KeyError) as e:
        raise ValueError("Dataset information is missing or invalid") from e


@timed("Get upload access credentials")
def get_upload_credentials(dataset_name: str, cfg: Optional[Config] = None) -> Optional[ResourceCredentials]:
    """Get dataset details by name from the Hafnia platform."""
    cfg = cfg or Config()
    dataset_response = get_dataset_by_name(dataset_name=dataset_name, cfg=cfg)
    if dataset_response is None:
        return None

    return get_upload_credentials_by_id(dataset_response["id"], cfg=cfg)


@timed("Get upload access credentials by ID")
def get_upload_credentials_by_id(dataset_id: str, cfg: Optional[Config] = None) -> Optional[ResourceCredentials]:
    """Get dataset details by ID from the Hafnia platform."""
    cfg = cfg or Config()

    endpoint_dataset = cfg.get_platform_endpoint("datasets")
    header = {"Authorization": cfg.api_key}
    full_url = f"{endpoint_dataset}/{dataset_id}/temporary-credentials-upload"
    credentials_response: Dict = http.fetch(full_url, headers=header)  # type: ignore[assignment]

    return ResourceCredentials.fix_naming(credentials_response)


@timed("Delete dataset by id")
def delete_dataset_by_id(dataset_id: str, cfg: Optional[Config] = None) -> Dict:
    cfg = cfg or Config()
    endpoint_dataset = cfg.get_platform_endpoint("datasets")
    header = {"Authorization": cfg.api_key}
    full_url = f"{endpoint_dataset}/{dataset_id}"
    return http.delete(full_url, headers=header)  # type: ignore


@timed("Delete dataset by name")
def delete_dataset_by_name(dataset_name: str, cfg: Optional[Config] = None) -> Dict:
    cfg = cfg or Config()
    dataset_response = get_dataset_by_name(dataset_name=dataset_name, cfg=cfg)
    if dataset_response is None:
        raise ValueError(f"Dataset '{dataset_name}' not found on the Hafnia platform.")

    dataset_id = dataset_response["id"]  # type: ignore[union-attr]
    response = delete_dataset_by_id(dataset_id=dataset_id, cfg=cfg)
    user_logger.info(f"Dataset '{dataset_name}' has been deleted from the Hafnia platform.")
    return response


def delete_dataset_completely_by_name(dataset_name: str, interactive: bool = True) -> None:
    from hafnia.dataset.operations.dataset_s3_storage import delete_hafnia_dataset_files_on_platform

    cfg = Config()

    is_deleted = delete_hafnia_dataset_files_on_platform(
        dataset_name=dataset_name,
        interactive=interactive,
        cfg=cfg,
    )
    if not is_deleted:
        return
    delete_dataset_by_name(dataset_name, cfg=cfg)


@timed("Import dataset details to platform")
def upload_dataset_details(cfg: Config, data: dict, dataset_name: str) -> dict:
    dataset_endpoint = cfg.get_platform_endpoint("datasets")
    dataset_id = get_dataset_id(dataset_name, dataset_endpoint, cfg.api_key)

    import_endpoint = f"{dataset_endpoint}/{dataset_id}/import"
    headers = {"Authorization": cfg.api_key}

    user_logger.info("Exporting dataset details to platform. This may take up to 30 seconds...")
    response = post(endpoint=import_endpoint, headers=headers, data=data)  # type: ignore[assignment]
    return response  # type: ignore[return-value]


def download_or_get_dataset_path(
    dataset_name: str,
    cfg: Optional[Config] = None,
    path_datasets_folder: Optional[str] = None,
    force_redownload: bool = False,
    download_files: bool = True,
) -> Path:
    """Download or get the path of the dataset."""
    recipe_explicit = DatasetRecipe.from_implicit_form(dataset_name)
    path_dataset = get_dataset_path_from_recipe(recipe_explicit, path_datasets=path_datasets_folder)

    is_dataset_valid = HafniaDataset.check_dataset_path(path_dataset, raise_error=False)
    if is_dataset_valid and not force_redownload:
        user_logger.info("Dataset found locally. Set 'force=True' or add `--force` flag with cli to re-download")
        return path_dataset

    cfg = cfg or Config()
    api_key = cfg.api_key

    shutil.rmtree(path_dataset, ignore_errors=True)

    endpoint_dataset = cfg.get_platform_endpoint("datasets")
    dataset_res = get_dataset_by_name(dataset_name, cfg)  # Check if dataset exists
    if dataset_res is None:
        raise ValueError(f"Dataset '{dataset_name}' not found on the Hafnia platform.")

    dataset_id = dataset_res.get("id")  # type: ignore[union-attr]

    if utils.is_hafnia_cloud_job():
        credentials_endpoint_suffix = "temporary-credentials-hidden"  # Access to hidden datasets
    else:
        credentials_endpoint_suffix = "temporary-credentials"  # Access to sample dataset
    access_dataset_endpoint = f"{endpoint_dataset}/{dataset_id}/{credentials_endpoint_suffix}"

    download_dataset_from_access_endpoint(
        endpoint=access_dataset_endpoint,
        api_key=api_key,
        path_dataset=path_dataset,
        download_files=download_files,
    )
    return path_dataset


def download_dataset_from_access_endpoint(
    endpoint: str,
    api_key: str,
    path_dataset: Path,
    version: Optional[str] = None,
    download_files: bool = True,
) -> None:
    try:
        resource_credentials = get_resource_credentials(endpoint, api_key)
        download_annotation_dataset_from_version(
            version=version,
            credentials=resource_credentials,
            path_dataset=path_dataset,
        )

    except ValueError as e:
        user_logger.error(f"Failed to download annotations: {e}")
        return

    if not download_files:
        return
    dataset = HafniaDataset.from_path(path_dataset, check_for_images=False)
    try:
        dataset = dataset.download_files_aws(path_dataset, aws_credentials=resource_credentials, force_redownload=True)
    except ValueError as e:
        user_logger.error(f"Failed to download images: {e}")
        return
    dataset.write_annotations(path_folder=path_dataset)  # Overwrite annotations as files have been re-downloaded


TABLE_FIELDS = {
    "ID": "id",
    "Hidden\nSamples": "hidden.samples",
    "Hidden\nSize": "hidden.size",
    "Sample\nSamples": "sample.samples",
    "Sample\nSize": "sample.size",
    "Name": "name",
    "Title": "title",
}


def pretty_print_datasets(datasets: List[Dict[str, str]]) -> None:
    datasets = extend_dataset_details(datasets)
    datasets = sorted(datasets, key=lambda x: x["name"].lower())

    table = rich.table.Table(title="Available Datasets")
    for i_dataset, dataset in enumerate(datasets):
        if i_dataset == 0:
            for column_name, _ in TABLE_FIELDS.items():
                table.add_column(column_name, justify="left", style="cyan", no_wrap=True)
        row = [str(dataset.get(field, "")) for field in TABLE_FIELDS.values()]
        table.add_row(*row)

    rprint(table)


def extend_dataset_details(datasets: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Extends dataset details with number of samples and size"""
    for dataset in datasets:
        for variant in dataset["dataset_variants"]:
            variant_type = variant["variant_type"]
            dataset[f"{variant_type}.samples"] = variant["number_of_data_items"]
            dataset[f"{variant_type}.size"] = utils.size_human_readable(variant["size_bytes"])
    return datasets


def download_annotation_dataset_from_version(
    version: Optional[str],
    credentials: ResourceCredentials,
    path_dataset: Path,
) -> list[str]:
    path_dataset.mkdir(parents=True, exist_ok=True)

    envs = credentials.aws_credentials()
    bucket_prefix_sample_versions = f"{credentials.s3_uri()}/versions"
    all_s3_annotation_files = s5cmd_utils.list_bucket(bucket_prefix=bucket_prefix_sample_versions, append_envs=envs)
    s3_files = _annotation_files_from_version(version=version, all_annotation_files=all_s3_annotation_files)

    local_paths = [(path_dataset / filename.split("/")[-1]).as_posix() for filename in s3_files]
    s5cmd_utils.fast_copy_files(
        src_paths=s3_files,
        dst_paths=local_paths,
        append_envs=envs,
        description="Downloading annotation files",
    )
    return local_paths


def _annotation_files_from_version(version: Optional[str], all_annotation_files: list[str]) -> list[str]:
    version_files = collections.defaultdict(list)
    for metadata_file in all_annotation_files:
        version_str, filename = metadata_file.split("/")[-2:]
        if filename not in DATASET_FILENAMES_REQUIRED:
            continue
        version_files[version_str].append(metadata_file)
    available_versions = {v for v, files in version_files.items() if len(files) == len(DATASET_FILENAMES_REQUIRED)}

    if len(available_versions) == 0:
        raise ValueError("No versions were found in the dataset.")

    if version is None:
        latest_version = max(Version(ver) for ver in available_versions)
        version = str(latest_version)
        user_logger.info(f"No version selected. Using latest version: {version}")

    if version not in available_versions:
        raise ValueError(f"Selected version '{version}' not found in available versions: {available_versions}")

    return version_files[version]
