import subprocess
import sys
from pathlib import Path

from mdi_python_tools.log import logger
from mdi_python_tools.platform.runtime import collect_python_modules, handle_status


def handle_launch(task: str) -> None:
    """
    Launch and execute a specified MDI task.

    This function verifies the MDI environment status, locates the task script,
    and executes it in a subprocess with output streaming.

    Args:
        task (str): Name of the task to execute

    Raises:
        ValueError: If the task is not found or scripts directory is not in PYTHONPATH
    """

    status = handle_status()
    if status.status != "ok":
        logger.warning(f"MDI environment not ready: {status}")
        return
    scripts_dir = [p for p in sys.path if "scripts" in p][0]
    scripts = collect_python_modules(Path(scripts_dir))
    if task not in scripts:
        available_tasks = ", ".join(sorted(scripts.keys()))
        raise ValueError(f"Task '{task}' not found. Available tasks: {available_tasks}")
    subprocess.check_call(
        ["python", scripts[task]["runner_path"]], stdout=sys.stdout, stderr=sys.stdout
    )
