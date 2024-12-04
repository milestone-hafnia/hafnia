#!/usr/bin/env python

import os
import sys
from dataclasses import dataclass
from argparse import ArgumentParser
from pathlib import Path
from enum import Enum
from typing import Dict, Optional
import subprocess


class COMMAND(Enum):
    STATUS = "status"
    MOUNT_USER_SRC = "mount"
    LAUNCH = "launch"
    PREPARE = "prepare"


@dataclass
class MDIStatus:
    python_path: str = os.environ.get("PYTHONPATH", "")
    status: Optional[str] = None
    message: Optional[str] = None

    def __repr__(self) -> str:
        key_width = max(len(key) for key in self.__dict__.keys())
        value_width = max(
            len(str(value)) for value in self.__dict__.values() if value is not None
        )
        table_width = key_width + value_width + 7
        output = ["-" * table_width]
        for key in ("status", "python_path", "message"):
            value = self.__dict__[key] if self.__dict__[key] is not None else ""
            output.append(f"| {key:<{key_width}} | {value:<{value_width}} |")
        output.append("-" * table_width)

        return "\n".join(output)


def collect_python_modules(directory: Path) -> Dict:
    """
    Collect Python modules in the specified directory and its subdirectories,
    excluding files that start with '_' or '.'.

    Args:
        directory (str or Path): The directory to search in.

    Returns:
        dict: A dictionary where keys are task names and values are module details.
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
            "runner_path": module.__file__,
        }
        modules[task_name] = module_details

    return modules


def handle_status(verbose: int = 0) -> MDIStatus:
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
    status = handle_status()
    if status.status == "ok":
        print(status)
        return
    source = Path(source)
    lib_dir = source / "lib"
    scripts_dir = source / "scripts"
    if not lib_dir.exists() and not scripts_dir.exists():
        raise FileNotFoundError(
            f"Filestructure is not suppored. Expected 'lib' and 'scripts' directories in {source}."
        )
    cmd = f"export PYTHONPATH=$PYTHONPATH:{lib_dir}:{scripts_dir}"
    # Requiered for the shell execution
    print(cmd)
    sys.path.extend([lib_dir.as_posix(), scripts_dir.as_posix()])


def handle_prepare(source: str, exec_cmd: str) -> None:
    status = handle_status()
    if status.status != "ok":
        print(status)
        return

    scripts = collect_python_modules(Path(source) / "scripts")
    if exec_cmd not in scripts:
        raise ValueError(f"Task {exec_cmd} not found in {source}/scripts.")

    with open(f"{source}/entrypoint.sh", "w") as f:
        f.write("#!/bin/bash\n")
        f.write(f"eval $(mdi mount {source})\n")
        f.write(f"mdi launch {exec_cmd}\n")
    os.chmod(f"{source}/entrypoint.sh", 0o755)
    print(f"Entrypoint script created at {source}/entrypoint.sh")


def handle_launch(task: str) -> None:
    status = handle_status()
    if status.status != "ok":
        print(status)
        return
    scripts_dir = [p for p in sys.path if "scripts" in p][0]
    scripts = collect_python_modules(Path(scripts_dir))
    if task not in scripts:
        raise ValueError(f"Task {task} not found.")
    subprocess.check_call(
        ["python", scripts[task]["runner_path"]], stdout=sys.stdout, stderr=sys.stdout
    )


def create_parser() -> ArgumentParser:
    parser = ArgumentParser(description="MDI Run Contrainer cli for executing jobs")
    subparsers = parser.add_subparsers(dest="command", help="Sub-command to run")

    status_parser = subparsers.add_parser(COMMAND.STATUS.value, help="Check the status")
    status_parser.add_argument(
        "--verbose", type=int, default=1, help="Print verbose output"
    )
    status_parser.set_defaults(func=handle_status)

    mount_parser = subparsers.add_parser(
        COMMAND.MOUNT_USER_SRC.value, help="Mount user source"
    )
    mount_parser.add_argument("source", type=str, help="Source path to mount")
    mount_parser.set_defaults(func=handle_mount)

    prepare_parser = subparsers.add_parser(
        COMMAND.PREPARE.value, help="Prepare entrypoint."
    )
    prepare_parser.add_argument("source", type=str, help="Source path to the codebase.")
    prepare_parser.add_argument("exec_cmd", type=str, help="Command to run.")
    prepare_parser.set_defaults(func=handle_prepare)

    launch_parser = subparsers.add_parser(COMMAND.LAUNCH.value, help="Launch a job")
    launch_parser.add_argument("task", type=str, help="Name of the job to launch")
    launch_parser.set_defaults(func=handle_launch)

    return parser


def main():
    parser = create_parser()
    args = parser.parse_args()

    # Retrieve the function associated with the command and call it
    if hasattr(args, "func"):
        func = args.func
        func(**{k: v for k, v in vars(args).items() if k not in ["func", "command"]})
    else:
        parser.print_help()
