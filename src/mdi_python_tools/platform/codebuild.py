import os
import sys
from pathlib import Path
import boto3
from dataclasses import dataclass
from typing import Dict, Optional

from mdi_python_tools.log import logger
from mdi_python_tools.platform.common import get_credentials


@dataclass
class MDIStatus:
    python_path: str = os.environ.get("PYTHONPATH", "")
    status: Optional[str] = None
    message: Optional[str] = None

    def __repr__(self) -> str:
        key_width = max(len(key) for key in self.__dict__.keys())
        value_width = max(len(str(value)) for value in self.__dict__.values() if value is not None)
        table_width = key_width + value_width + 7
        output = ["-" * table_width]
        for key in ("status", "python_path", "message"):
            value = self.__dict__[key] if self.__dict__[key] is not None else ""
            output.append(f"| {key:<{key_width}} | {value:<{value_width}} |")
        output.append("-" * table_width)

        return "\n".join(output)


def handle_status(verbose: int = 0) -> MDIStatus:
    """
    Checks and returns the current status of the MDI environment.

    Args:
        verbose (int): If non-zero, prints the status to stdout

    Returns:
        MDIStatus: Object containing the current MDI environment status
    """
    status = MDIStatus()
    if "lib" not in status.python_path and "scripts" not in status.python_path:
        status.status = "error"
        status.message = "Use 'eval $(mdi mount <PATH>)'"
    else:
        status.status = "ok"
        status.message = "MDI is ready to run tasks."
    if verbose:
        print(status)
    return status


def handle_mount(source: str) -> None:
    """
    Mounts the MDI environment by adding source directories to PYTHONPATH.

    Args:
        source (str): Path to the root directory containing 'lib' and 'scripts' subdirectories

    Raises:
        FileNotFoundError: If the required directory structure is not found
    """
    status = handle_status()
    if status.status == "ok":
        print(status)
        return

    source_path = Path(source)
    lib_dir = source_path / "lib"
    scripts_dir = source_path / "scripts"

    if not lib_dir.exists() and not scripts_dir.exists():
        raise FileNotFoundError(
            f"Filestructure is not supported. Expected 'lib' and 'scripts' directories in {source_path}."
        )
    cmd = f"export PYTHONPATH=$PYTHONPATH:{lib_dir}:{scripts_dir}"
    # Required for the shell execution
    print(cmd)
    sys.path.extend([lib_dir.as_posix(), scripts_dir.as_posix()])


def collect_python_modules(directory: Path) -> Dict[str, Dict[str, str]]:
    """
    Collects Python modules from a directory and its subdirectories.

    This function dynamically imports Python modules found in the specified directory,
    excluding files that start with '_' or '.'. It's used to discover available tasks
    in the MDI environment.

    Args:
        directory (Path): The directory to search for Python modules

    Returns:
        Dict[str, Dict[str, str]]: A dictionary mapping task names to module details, where each detail contains:
            - module_name (str): The full module name
            - runner_path (str): The absolute path to the module file
    """
    from importlib.util import spec_from_file_location, module_from_spec

    modules = {}
    for fname in directory.rglob("*.py"):
        if fname.name.startswith("-"):
            continue

        task_name = fname.stem
        module_name = f"{directory.name}.{task_name}"

        # Import the module dynamically
        spec = spec_from_file_location(module_name, fname)
        module = module_from_spec(spec)
        spec.loader.exec_module(module)

        module_details = {
            "module_name": module_name,
            "runner_path": str(fname.resolve()),
        }
        modules[task_name] = module_details

    return modules


def handle_prepare(source: str, exec_cmd: str) -> None:
    """
    Prepares an entrypoint script for executing a specific MDI task.

    This function validates the environment status and creates a shell script
    that properly initializes the MDI environment before running the specified task.

    Args:
        source (str): Path to the MDI source directory
        exec_cmd (str): Name of the task to execute

    Raises:
        ValueError: If the specified task is not found in the scripts directory
    """
    status = handle_status()
    if status.status != "ok":
        logger.warning(status)
        return

    source_path = Path(source)
    scripts = collect_python_modules(source_path / "scripts")
    if exec_cmd not in scripts:
        raise ValueError(f"Task {exec_cmd} not found in {source_path/'scripts'}.")

    entrypoint_path = source_path / "entrypoint.sh"
    entrypoint_path.write_text(
        "#!/bin/bash\n" f"eval $(mdi mount {source_path})\n" f"mdi launch {exec_cmd}\n"
    )
    entrypoint_path.chmod(0o755)
    logger.info(f"Entrypoint script created at {source}/entrypoint.sh")


def get_recipe(recipe_url: str, aws_region: str, output_dir: str) -> Path:
    """
    Downloads a recipe file from AWS S3 using temporary credentials.

    This function retrieves credentials using the recipe URL, validates the S3 URI,
    and downloads the recipe file to the specified output directory.

    Args:
        recipe_url (str): URL to retrieve AWS credentials
        aws_region (str): AWS region where the S3 bucket is located
        output_dir (str): Local directory to save the downloaded recipe

    Returns:
        Path: Path object pointing to the downloaded recipe file

    Raises:
        ValueError: If the S3 URI format is invalid
        RuntimeError: If the download operation fails
    """
    credentials = get_credentials(recipe_url, aws_region)
    arn_prefix = "arn:aws:s3:::"
    uri = credentials["s3_path"]
    if not uri.startswith(arn_prefix):
        raise ValueError(f"Invalid S3 URI: {uri}")
    uri = uri[len(arn_prefix) :]

    with boto3.client(
        "s3",
        aws_access_key_id=credentials["access_key"],
        aws_secret_access_key=credentials["secret_key"],
        aws_session_token=credentials["session_token"],
    ) as recipe_s3:
        bucket_name, *key_parts = uri.split("/")
        key = "/".join(key_parts)

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        file_path = output_path / Path(key).name

        try:
            recipe_s3.download_file(bucket_name, key, str(file_path))
            logger.info(f"Recipe downloaded successfully to {file_path}")
        except Exception as e:
            if file_path.exists():
                file_path.unlink()
            raise RuntimeError(
                f"Failed to download recipe: {e} from bucket {bucket_name} with key {key}"
            )
        return file_path
