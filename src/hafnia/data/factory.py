import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

from tqdm import tqdm

from cli.config import Config
from hafnia import utils
from hafnia.dataset import dataset_names
from hafnia.dataset.hafnia_dataset import HafniaDataset
from hafnia.log import user_logger
from hafnia.platform import get_dataset_id
from hafnia.platform.download import get_resource_credentials


def load_dataset(dataset_name: str, force_redownload: bool = False) -> HafniaDataset:
    """Load a dataset either from a local path or from the Hafnia platform."""

    path_dataset = get_dataset_path(dataset_name, force_redownload=force_redownload)
    dataset = HafniaDataset.read_from_path(path_dataset)
    return dataset


def get_dataset_path(dataset_name: str, force_redownload: bool = False) -> Path:
    path_dataset = download_or_get_dataset_path(
        dataset_name=dataset_name,
        force_redownload=force_redownload,
    )
    return path_dataset


def download_or_get_dataset_path(
    dataset_name: str,
    cfg: Optional[Config] = None,
    output_dir: Optional[str] = None,
    force_redownload: bool = False,
) -> Path:
    """Download or get the path of the dataset."""
    if utils.is_remote_job():
        return Path(os.getenv("MDI_DATASET_DIR", "/opt/ml/input/data/training"))

    cfg = cfg or Config()
    endpoint_dataset = cfg.get_platform_endpoint("datasets")
    api_key = cfg.api_key

    output_dir = output_dir or str(utils.PATH_DATASETS)
    dataset_path_sample = Path(output_dir).absolute() / dataset_name

    is_dataset_valid = HafniaDataset.check_dataset_path(dataset_path_sample, raise_error=False)
    if is_dataset_valid and not force_redownload:
        user_logger.info("Dataset found locally. Set 'force=True' or add `--force` flag with cli to re-download")
        return dataset_path_sample

    dataset_path_sample.mkdir(exist_ok=True, parents=True)
    dataset_id = get_dataset_id(dataset_name, endpoint_dataset, api_key)
    dataset_access_info_url = f"{endpoint_dataset}/{dataset_id}/temporary-credentials"

    if force_redownload and dataset_path_sample.exists():
        # Remove old files to avoid old files conflicting with new files
        shutil.rmtree(dataset_path_sample, ignore_errors=True)

    res_creds = get_resource_credentials(dataset_access_info_url, api_key)
    s3_arn = res_creds["s3_path"]
    arn_prefix = "arn:aws:s3:::"
    s3_uri = s3_arn.replace(arn_prefix, "s3://")

    envs = {
        "AWS_ACCESS_KEY_ID": res_creds["access_key"],
        "AWS_SECRET_ACCESS_KEY": res_creds["secret_key"],
        "AWS_SESSION_TOKEN": res_creds["session_token"],
    }

    s3_dataset_files = [f"{s3_uri}/{filename}" for filename in dataset_names.DATASET_FILENAMES]
    local_dataset_paths = [str(dataset_path_sample / filename) for filename in dataset_names.DATASET_FILENAMES]
    fast_copy_files_s3(
        src_paths=s3_dataset_files,
        dst_paths=local_dataset_paths,
        append_envs=envs,
        description="Downloading annotations",
    )

    dataset = HafniaDataset.read_from_path(dataset_path_sample, check_for_images=False)
    fast_copy_files_s3(
        src_paths=dataset.samples[dataset_names.ColumnName.REMOTE_PATH].to_list(),
        dst_paths=dataset.samples[dataset_names.ColumnName.FILE_NAME].to_list(),
        append_envs=envs,
        description="Downloading images",
    )

    return dataset_path_sample


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
        lines = []
        for line in tqdm(process.stdout, total=len(commands), desc=description):
            lines.append(line.strip())
    return lines
