import os
import shutil
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import rich
from rich import print as rprint
from rich.progress import track

from cli.config import Config
from hafnia import http, utils
from hafnia.dataset.dataset_names import DATASET_FILENAMES_REQUIRED, ColumnName
from hafnia.dataset.dataset_recipe.dataset_recipe import (
    DatasetRecipe,
    get_dataset_path_from_recipe,
)
from hafnia.dataset.hafnia_dataset import HafniaDataset
from hafnia.http import fetch
from hafnia.log import sys_logger, user_logger
from hafnia.platform.download import get_resource_credentials
from hafnia.utils import timed


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
    dataset_id = get_dataset_id(dataset_name=dataset_name, endpoint=endpoint_dataset, api_key=api_key)
    if dataset_id is None:
        sys_logger.error(f"Dataset '{dataset_name}' not found on the Hafnia platform.")

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
    download_files: bool = True,
) -> None:
    resource_credentials = get_resource_credentials(endpoint, api_key)

    local_dataset_paths = [(path_dataset / filename).as_posix() for filename in DATASET_FILENAMES_REQUIRED]
    s3_uri = resource_credentials.s3_uri()
    s3_dataset_files = [f"{s3_uri}/{filename}" for filename in DATASET_FILENAMES_REQUIRED]

    envs = resource_credentials.aws_credentials()
    try:
        fast_copy_files_s3(
            src_paths=s3_dataset_files,
            dst_paths=local_dataset_paths,
            append_envs=envs,
            description="Downloading annotations",
        )
    except ValueError as e:
        user_logger.error(f"Failed to download annotations: {e}")
        return

    if not download_files:
        return
    dataset = HafniaDataset.from_path(path_dataset, check_for_images=False)
    try:
        fast_copy_files_s3(
            src_paths=dataset.samples[ColumnName.REMOTE_PATH].to_list(),
            dst_paths=dataset.samples[ColumnName.FILE_PATH].to_list(),
            append_envs=envs,
            description="Downloading images",
        )
    except ValueError as e:
        user_logger.error(f"Failed to download images: {e}")
        return


def fast_copy_files_s3(
    src_paths: List[str],
    dst_paths: List[str],
    append_envs: Optional[Dict[str, str]] = None,
    description: str = "Copying files",
) -> List[str]:
    if len(src_paths) != len(dst_paths):
        raise ValueError("Source and destination paths must have the same length.")
    cmds = [f"cp {src} {dst}" for src, dst in zip(src_paths, dst_paths)]
    lines = execute_s5cmd_commands(cmds, append_envs=append_envs, description=description)
    return lines


def find_s5cmd() -> Optional[str]:
    """Locate the s5cmd executable across different installation methods.

    Searches for s5cmd in:
    1. System PATH (via shutil.which)
    2. Python bin directory (Unix-like systems)
    3. Python executable directory (direct installs)

    Returns:
        str: Absolute path to s5cmd executable if found, None otherwise.
    """
    result = shutil.which("s5cmd")
    if result:
        return result
    python_dir = Path(sys.executable).parent
    locations = (python_dir / "Scripts" / "s5cmd.exe", python_dir / "bin" / "s5cmd", python_dir / "s5cmd")
    for loc in locations:
        if loc.exists():
            return str(loc)
    return None


def execute_s5cmd_commands(
    commands: List[str],
    append_envs: Optional[Dict[str, str]] = None,
    description: str = "Executing s5cmd commands",
) -> List[str]:
    append_envs = append_envs or {}
    # In Windows default "Temp" directory can not be deleted that is why we need to create a
    # temporary directory.
    with tempfile.TemporaryDirectory() as temp_dir:
        tmp_file_path = Path(temp_dir, f"{uuid.uuid4().hex}.txt")
        tmp_file_path.write_text("\n".join(commands))

        s5cmd_bin = find_s5cmd()
        if s5cmd_bin is None:
            raise ValueError("Can not find s5cmd executable.")
        run_cmds = [s5cmd_bin, "run", str(tmp_file_path)]
        sys_logger.debug(run_cmds)
        envs = os.environ.copy()
        envs.update(append_envs)

        process = subprocess.Popen(
            run_cmds,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            env=envs,
        )

        error_lines = []
        lines = []
        for line in track(process.stdout, total=len(commands), description=description):
            if "ERROR" in line or "error" in line:
                error_lines.append(line.strip())
            lines.append(line.strip())

        if len(error_lines) > 0:
            show_n_lines = min(5, len(error_lines))
            str_error_lines = "\n".join(error_lines[:show_n_lines])
            user_logger.error(
                f"Detected {len(error_lines)} errors occurred while executing a total of {len(commands)} "
                f" commands with s5cmd. The first {show_n_lines} is printed below:\n{str_error_lines}"
            )
            raise RuntimeError("Errors occurred during s5cmd execution.")
    return lines


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
