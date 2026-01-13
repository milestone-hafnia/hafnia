import os
import shutil
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path
from typing import Dict, List, Optional

from hafnia.log import sys_logger, user_logger
from hafnia.utils import progress_bar


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
    locations = (
        python_dir / "Scripts" / "s5cmd.exe",
        python_dir / "bin" / "s5cmd",
        python_dir / "s5cmd",
    )
    for loc in locations:
        if loc.exists():
            return str(loc)
    return None


def execute_command(args: List[str], append_envs: Optional[Dict[str, str]] = None) -> subprocess.CompletedProcess:
    s5cmd_bin = find_s5cmd()
    cmds = [s5cmd_bin] + args
    envs = os.environ.copy()
    if append_envs:
        envs.update(append_envs)

    result = subprocess.run(
        cmds,  # type: ignore[arg-type]
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
        env=envs,
    )
    return result


def execute_commands(
    commands: List[str],
    append_envs: Optional[Dict[str, str]] = None,
    description: str = "Executing s5cmd commands",
) -> List[str]:
    append_envs = append_envs or {}

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
        for line in progress_bar(process.stdout, total=len(commands), description=description):  # type: ignore[arg-type]
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


def delete_bucket_content(
    bucket_prefix: str,
    remove_bucket: bool = True,
    append_envs: Optional[Dict[str, str]] = None,
) -> None:
    # Remove all files in the bucket
    returns = execute_command(["rm", f"{bucket_prefix}/*"], append_envs=append_envs)

    if returns.returncode != 0:
        bucket_content_is_already_deleted = "no object found" in returns.stderr.strip()
        bucket_is_already_deleted = "NoSuchBucket" in returns.stderr.strip()
        if bucket_content_is_already_deleted:
            user_logger.info(f"No action was taken. S3 bucket '{bucket_prefix}' is already empty.")
        elif bucket_is_already_deleted:
            user_logger.info(f"No action was taken. S3 bucket '{bucket_prefix}' does not exist.")
            return
        else:
            user_logger.error("Error during s5cmd rm command:")
            user_logger.error(returns.stdout)
            user_logger.error(returns.stderr)
            raise RuntimeError(f"Failed to delete all files in S3 bucket '{bucket_prefix}'.")

    if remove_bucket:
        # Remove the bucket itself
        returns = execute_command(["rb", bucket_prefix], append_envs=append_envs)
        if returns.returncode != 0:
            user_logger.error("Error during s5cmd rb command:")
            user_logger.error(returns.stdout)
            user_logger.error(returns.stderr)
            raise RuntimeError(f"Failed to delete S3 bucket '{bucket_prefix}'.")
    user_logger.info(f"S3 bucket '{bucket_prefix}' has been deleted.")


def list_bucket(bucket_prefix: str, append_envs: Optional[Dict[str, str]] = None) -> List[str]:
    output = execute_command(["ls", f"{bucket_prefix}/*"], append_envs=append_envs)
    has_missing_folder = "no object found" in output.stderr.strip()
    if output.returncode != 0 and not has_missing_folder:
        user_logger.error("Error during s5cmd ls command:")
        user_logger.error(output.stderr)
        raise RuntimeError(f"Failed to list dataset in S3 bucket '{bucket_prefix}'.")

    files_in_s3 = [f"{bucket_prefix}/{line.split(' ')[-1]}" for line in output.stdout.splitlines()]
    return files_in_s3


def fast_copy_files(
    src_paths: List[str],
    dst_paths: List[str],
    append_envs: Optional[Dict[str, str]] = None,
    description: str = "Copying files",
) -> List[str]:
    if len(src_paths) != len(dst_paths):
        raise ValueError("Source and destination paths must have the same length.")
    cmds = [f"cp {src} {dst}" for src, dst in zip(src_paths, dst_paths)]
    lines = execute_commands(cmds, append_envs=append_envs, description=description)
    return lines
