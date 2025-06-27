import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import rich
from tqdm import tqdm

from cli.config import Config
from hafnia import utils
from hafnia.dataset import dataset_names
from hafnia.dataset.hafnia_dataset import HafniaDataset
from hafnia.http import fetch
from hafnia.log import user_logger
from hafnia.platform import get_dataset_id
from hafnia.platform.download import get_resource_credentials
from hafnia.utils import timed


@timed("Fetching dataset list.")
def dataset_list(cfg: Optional[Config] = None) -> List[Dict[str, str]]:
    """List available datasets on the Hafnia platform."""
    cfg = cfg or Config()
    endpoint_dataset = cfg.get_platform_endpoint("datasets")
    header = {"Authorization": cfg.api_key}
    datasets: List[Dict[str, str]] = fetch(endpoint_dataset, headers=header)  # type: ignore
    if not datasets:
        raise ValueError("No datasets found on the Hafnia platform.")

    return datasets


def download_or_get_dataset_path(
    dataset_name: str,
    cfg: Optional[Config] = None,
    path_datasets_folder: Optional[str] = None,
    force_redownload: bool = False,
) -> Path:
    """Download or get the path of the dataset."""
    if utils.is_remote_job():
        return Path(os.getenv("MDI_DATASET_DIR", "/opt/ml/input/data/training"))

    path_datasets_folder = path_datasets_folder or str(utils.PATH_DATASETS)
    path_dataset = Path(path_datasets_folder).absolute() / dataset_name

    is_dataset_valid = HafniaDataset.check_dataset_path(path_dataset, raise_error=False)
    if is_dataset_valid and not force_redownload:
        user_logger.info("Dataset found locally. Set 'force=True' or add `--force` flag with cli to re-download")
        return path_dataset

    cfg = cfg or Config()
    api_key = cfg.api_key

    shutil.rmtree(path_dataset, ignore_errors=True)

    endpoint_dataset = cfg.get_platform_endpoint("datasets")
    dataset_id = get_dataset_id(dataset_name=dataset_name, endpoint=endpoint_dataset, api_key=api_key)
    access_dataset_endpoint = f"{endpoint_dataset}/{dataset_id}/temporary-credentials"

    download_dataset_from_access_endpoint(
        endpoint=access_dataset_endpoint,
        api_key=api_key,
        path_dataset=path_dataset,
    )
    return path_dataset


def download_dataset_from_access_endpoint(endpoint: str, api_key: str, path_dataset: Path) -> None:
    resource_credentials = get_resource_credentials(endpoint, api_key)

    local_dataset_paths = [str(path_dataset / filename) for filename in dataset_names.DATASET_FILENAMES]
    s3_uri = resource_credentials.s3_uri()
    s3_dataset_files = [f"{s3_uri}/{filename}" for filename in dataset_names.DATASET_FILENAMES]

    envs = resource_credentials.aws_credentials()
    fast_copy_files_s3(
        src_paths=s3_dataset_files,
        dst_paths=local_dataset_paths,
        append_envs=envs,
        description="Downloading annotations",
    )

    dataset = HafniaDataset.read_from_path(path_dataset, check_for_images=False)
    fast_copy_files_s3(
        src_paths=dataset.samples[dataset_names.ColumnName.REMOTE_PATH].to_list(),
        dst_paths=dataset.samples[dataset_names.ColumnName.FILE_NAME].to_list(),
        append_envs=envs,
        description="Downloading images",
    )


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


def execute_s5cmd_commands(
    commands: List[str],
    append_envs: Optional[Dict[str, str]] = None,
    description: str = "Executing s5cmd commands",
) -> List[str]:
    append_envs = append_envs or {}
    with tempfile.NamedTemporaryFile(suffix=".txt") as tmp_file:
        tmp_file_path = Path(tmp_file.name)
        tmp_file_path.write_text("\n".join(commands))
        run_cmds = [
            "s5cmd",
            "run",
            str(tmp_file_path),
        ]
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
        for line in tqdm(process.stdout, total=len(commands), desc=description):
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


def create_rich_table_from_dataset(datasets: List[Dict[str, str]]) -> rich.table.Table:
    datasets = extend_dataset_details(datasets)
    datasets = sorted(datasets, key=lambda x: x["name"].lower())

    table = rich.table.Table(title="Available Datasets")
    for i_dataset, dataset in enumerate(datasets):
        if i_dataset == 0:
            for column_name, _ in TABLE_FIELDS.items():
                table.add_column(column_name, justify="left", style="cyan", no_wrap=True)
        row = [str(dataset.get(field, "")) for field in TABLE_FIELDS.values()]
        table.add_row(*row)

    return table


def extend_dataset_details(datasets: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Extends dataset details with number of samples and size"""
    for dataset in datasets:
        for variant in dataset["dataset_variants"]:
            variant_type = variant["variant_type"]
            dataset[f"{variant_type}.samples"] = variant["number_of_data_items"]
            dataset[f"{variant_type}.size"] = utils.size_human_readable(variant["size_bytes"])
    return datasets
