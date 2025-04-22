import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from hafnia.platform.executor import (
    PythonModule,
    collect_python_modules,
    handle_launch,
    handle_mount,
)


@pytest.fixture
def recipe_dir(tmp_path: Path) -> Path:
    """Create a temporary recipe directory structure for testing."""
    src = tmp_path / "src"
    scripts = src / "scripts"
    scripts.mkdir(exist_ok=True, parents=True)
    with open(scripts / "train.py", "w") as f:
        f.write("print('Run training')")
    libs = src / "libs"
    libs.mkdir(exist_ok=True, parents=True)
    with open(libs / "lib.py", "w") as f:
        f.write("print('Lib data 1')")
    with open(libs / "lib2.py", "w") as f:
        f.write("print('Lib data 2')")

    return src


def test_successful_mount(recipe_dir: Path) -> None:
    """Test that handle_mount correctly adds paths to PYTHONPATH."""
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("PYTHONPATH", "")
        handle_mount(recipe_dir)

        lib_path = recipe_dir / "lib"
        scripts_path = recipe_dir / "scripts"
        assert lib_path.as_posix() in sys.path
        assert scripts_path.as_posix() in sys.path


def test_successful_task_execution(recipe_dir: Path):
    """Test successful execution of a task."""
    mock_subprocess = MagicMock()
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("RECIPE_DIR", str(recipe_dir))
        mp.setattr(sys, "path", [str(recipe_dir / "scripts")])
        mp.setattr("hafnia.platform.executor.subprocess.check_call", mock_subprocess)

        handle_launch("train")
        mock_subprocess.assert_called_once()
        args, kwargs = mock_subprocess.call_args
        cmd = args[0]
        assert "python" == cmd[0]
        assert "train.py" in cmd[1]


def test_failed_task_execution(recipe_dir: Path):
    """Test handling of task execution errors."""
    import subprocess

    mock_subprocess = MagicMock(side_effect=subprocess.CalledProcessError(1, "python"))
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("RECIPE_DIR", str(recipe_dir))
        mp.setattr(sys, "path", [str(recipe_dir / "scripts")])
        mp.setattr("hafnia.platform.executor.subprocess.check_call", mock_subprocess)
        with pytest.raises(SystemExit):
            handle_launch("train")


def test_collect_valid_modules(recipe_dir: Path) -> None:
    """Test that collect_python_modules finds all valid modules."""
    scripts_dir = recipe_dir / "scripts"
    modules = collect_python_modules(scripts_dir)

    assert "train" in modules

    train_module = modules["train"]
    assert isinstance(train_module, PythonModule)
    assert train_module.module_name == "scripts.train"
    assert "train.py" in train_module.runner_path
