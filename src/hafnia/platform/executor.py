import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

from hafnia.log import logger


@dataclass
class PythonModule:
    """Dataclass to store Python module details."""

    module_name: str
    runner_path: str


def handle_mount(source: str) -> None:
    """
    Mounts the Hafnia environment by adding source directories to PYTHONPATH.

    Args:
        source (str): Path to the root directory containing 'lib' and 'scripts' subdirectories

    Raises:
        FileNotFoundError: If the required directory structure is not found
    """
    source_path = Path(source)
    lib_dir = source_path / "lib"
    scripts_dir = source_path / "scripts"

    if not lib_dir.exists() and not scripts_dir.exists():
        logger.error(f"Filestructure is not supported. Expected 'lib' and 'scripts' directories in {source_path}.")
        exit(1)

    sys.path.extend([lib_dir.as_posix(), scripts_dir.as_posix()])
    python_path = os.getenv("PYTHONPATH", "")
    os.environ["PYTHONPATH"] = f"{python_path}:{lib_dir.as_posix()}:{scripts_dir.as_posix()}"
    logger.info(f"Mounted codebase from {source_path}")


def collect_python_modules(directory: Path) -> Dict[str, PythonModule]:
    """
    Collects Python modules from a directory and its subdirectories.

    This function dynamically imports Python modules found in the specified directory,
    excluding files that start with '_' or '.'. It's used to discover available tasks
    in the Hafnia environment.

    Args:
        directory (Path): The directory to search for Python modules

    Returns:
        Dict[str, Dict[str, str]]: A dictionary mapping task names to module details, where each detail contains:
            - module_name (str): The full module name
            - runner_path (str): The absolute path to the module file
    """
    from importlib.util import module_from_spec, spec_from_file_location

    modules = {}
    for fname in directory.rglob("*.py"):
        if fname.name.startswith("-"):
            continue

        task_name = fname.stem
        module_name = f"{directory.name}.{task_name}"

        spec = spec_from_file_location(module_name, fname)
        if spec is None:
            logger.warning(f"Was not able to load {module_name} from {fname}")
            continue
        if spec.loader is None:
            logger.warning(f"Loader is None for {module_name} from {fname}")
            continue
        module = module_from_spec(spec)
        spec.loader.exec_module(module)

        modules[task_name] = PythonModule(module_name, str(fname.resolve()))

    return modules


def handle_launch(task: str) -> None:
    """
    Launch and execute a specified Hafnia task.

    This function verifies the Hafnia environment status, locates the task script,
    and executes it in a subprocess with output streaming.

    Args:
        task (str): Name of the task to execute

    Raises:
        ValueError: If the task is not found or scripts directory is not in PYTHONPATH
    """
    recipe_dir = os.getenv("RECIPE_DIR", None)
    if recipe_dir is None:
        raise ValueError("RECIPE_DIR environment variable not set.")
    handle_mount(recipe_dir)
    scripts_dir = [p for p in sys.path if "scripts" in p][0]
    scripts = collect_python_modules(Path(scripts_dir))
    if task not in scripts:
        available_tasks = ", ".join(sorted(scripts.keys()))
        logger.error(f"Task '{task}' not found. Available tasks: {available_tasks}")
        exit(1)
    try:
        subprocess.check_call(["python", scripts[task].runner_path], stdout=sys.stdout, stderr=sys.stdout)
    except subprocess.CalledProcessError as e:
        logger.error(f"Error executing task: {str(e)}")
        exit(1)
