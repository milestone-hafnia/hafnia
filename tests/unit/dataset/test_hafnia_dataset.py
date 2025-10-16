import os
from pathlib import Path

import pytest
from packaging.version import Version

import hafnia
from hafnia.dataset.dataset_names import ColumnName, DeploymentStage
from hafnia.dataset.dataset_upload_helper import dataset_info_from_dataset
from hafnia.dataset.hafnia_dataset import DatasetInfo, HafniaDataset, Sample, TaskInfo

# from data_management import utils
from hafnia.dataset.operations import dataset_stats, dataset_transformations
from hafnia.dataset.primitives.classification import Classification
from tests.helper_testing import (
    get_hafnia_functions_from_module,
    get_path_micro_hafnia_dataset,
)


def test_dataset_info_serializing_deserializing(tmp_path: Path):
    # Create a sample dataset
    tasks = [
        TaskInfo(name="Sample Task", class_names=["Class A", "Class B"], primitive=Classification),
        TaskInfo(name="Another Task", class_names=["Class C", "Class D"], primitive=Classification),
    ]
    dataset_info = DatasetInfo(dataset_name="Sample Dataset", version="1.0", tasks=tasks)

    path_dataset = tmp_path / "example_dataset_info.json"
    dataset_info.write_json(path_dataset)

    # Deserialize from JSON
    loaded_dataset_info = DatasetInfo.from_json_file(path_dataset)
    assert loaded_dataset_info == dataset_info


def test_hafnia_dataset_save_and_load(tmp_path: Path):
    # Create a sample dataset
    task_info = TaskInfo(name="Sample Task", class_names=["Class A", "Class B"], primitive=Classification)
    dataset_info = DatasetInfo(
        dataset_name="Sample Dataset", version="1.0", tasks=[task_info], primitive=Classification
    )

    path_dataset = tmp_path / "test_hafnia_dataset"

    path_files = [path_dataset / "data" / f"video{i}.mp4" for i in range(2)]
    for path_file in path_files:
        path_file.parent.mkdir(parents=True, exist_ok=True)
        path_file.write_text("")

    samples = [
        Sample(
            file_path=str(path),
            height=100,
            width=200,
            split="train",
            classifications=[Classification(class_name="Class A", class_idx=0)],
        )
        for path in path_files
    ]
    dataset = HafniaDataset.from_samples_list(samples_list=samples, info=dataset_info)
    dataset.write(path_dataset, drop_null_cols=False)

    dataset_reloaded = HafniaDataset.from_path(path_dataset)
    assert dataset_reloaded.info == dataset.info
    table_expected = dataset.samples.drop(ColumnName.FILE_PATH)
    table_actual = dataset_reloaded.samples.drop(ColumnName.FILE_PATH)
    assert table_expected.equals(table_actual), "The samples tables do not match after reloading the dataset."


@pytest.mark.parametrize("function_name", get_hafnia_functions_from_module(dataset_transformations))
def test_hafnia_dataset_has_all_dataset_transforms(function_name: str):
    if function_name.startswith("_"):
        pytest.skip("Skipping private functions")
    module_filename = os.sep.join(Path(dataset_transformations.__file__).parts[-2:])
    module_stem = dataset_transformations.__name__.split(".")[-1]
    assert hasattr(HafniaDataset, function_name), (
        f"HafniaDataset expect that all functions in '{module_filename}' also exists as methods in HafniaDataset.\n"
        f"Function '{function_name}' is missing in HafniaDataset.\n"
        f"Please add '{function_name} = {module_stem}.{function_name}' to HafniaDataset class."
    )


@pytest.mark.parametrize("function_name", get_hafnia_functions_from_module(dataset_stats))
def test_hafnia_dataset_has_all_dataset_stats_functions(function_name: str):
    module_filename = os.sep.join(Path(dataset_stats.__file__).parts[-2:])
    module_stem = dataset_stats.__name__.split(".")[-1]
    assert hasattr(HafniaDataset, function_name), (
        f"HafniaDataset expect that all functions in '{module_filename}' also exists as methods in HafniaDataset.\n"
        f"Function '{function_name}' is missing in HafniaDataset.\n"
        f"Please add '{function_name} = {module_stem}.{function_name}' to HafniaDataset class."
    )


def test_dataset_info_from_dataset():
    dataset_name = "micro-tiny-dataset"
    path_dataset = get_path_micro_hafnia_dataset(dataset_name=dataset_name, force_update=False)
    dataset = HafniaDataset.from_path(path_dataset)
    dataset_info = dataset_info_from_dataset(
        dataset=dataset,
        deployment_stage=DeploymentStage.STAGING,
        path_sample=path_dataset,
        path_hidden=None,
    )

    # Check if dataset info can be serialized to JSON
    dataset_info_json = dataset_info.model_dump_json()  # noqa: F841


def test_task_info_validation_exceptions():
    with pytest.raises(ValueError, match="Class names must be unique"):
        TaskInfo(
            primitive=Classification,
            class_names=["car", "person", "car"],  # <-- Duplicate name is used
        )
    primitive_wrong_name = "WrongPrimitiveName"
    with pytest.raises(ValueError, match=f"Primitive '{primitive_wrong_name}' is not recognized."):
        TaskInfo(
            primitive=primitive_wrong_name,
            class_names=["person", "car"],  # <-- Duplicate name is used
        )


def test_dataset_info_validation_exceptions():
    # Use case 1: Same primitive - different task name is allowed!
    DatasetInfo(
        dataset_name="test_dataset",
        version="1.0",
        tasks=[
            TaskInfo(primitive=Classification, class_names=["car", "person"]),
            TaskInfo(primitive=Classification, class_names=["car", "person"], name="Task2"),
        ],
    )

    # Use case 2: Same primitive and same task name is NOT allowed
    with pytest.raises(ValueError, match="Tasks must be unique"):
        DatasetInfo(
            dataset_name="test_dataset",
            version="1.0",
            tasks=[
                TaskInfo(primitive=Classification, class_names=["car", "person"], name="my_task"),
                TaskInfo(primitive=Classification, class_names=["car", "person"], name="my_task"),
            ],
        )


def test_dataset_info_replace_task():
    task1 = TaskInfo(primitive=Classification, class_names=["car", "person"], name="Task1")
    task2 = TaskInfo(primitive=Classification, class_names=["cat", "dog"], name="Task2")
    dataset_info = DatasetInfo(
        dataset_name="test_dataset",
        version="1.0",
        tasks=[task1, task2],
    )

    # Create a new task to replace task1
    new_task1 = TaskInfo(primitive=Classification, class_names=["bus", "truck"], name="Task3")

    # Replace task1 with new_task1
    dataset_info_updated = dataset_info.replace_task(old_task=task1, new_task=new_task1)

    # Verify that the task has been updated
    assert len(dataset_info_updated.tasks) == 2
    assert new_task1 in dataset_info_updated.tasks
    assert task2 in dataset_info_updated.tasks
    assert task1 not in dataset_info_updated.tasks

    # Attempt to replace a non-existing task
    non_existing_task = TaskInfo(primitive=Classification, class_names=["bike"], name="NonExistingTask")
    with pytest.raises(ValueError, match="Task '.*' not found in dataset info."):
        dataset_info.replace_task(old_task=non_existing_task, new_task=new_task1)


def test_dataset_version_validation():
    # Valid version
    DatasetInfo(dataset_name="test_dataset", version="1.0")

    # Invalid version
    with pytest.raises(ValueError, match="Invalid dataset_version '.*'. Must be a valid version string"):
        DatasetInfo(dataset_name="test_dataset", version="invalid_version")


def test_dataset_format_version_validation():
    # Valid dataset format version
    dataset_info = DatasetInfo(dataset_name="test_dataset")
    assert dataset_info.format_version == hafnia.__dataset_format_version__

    # Explicitly set valid version
    dataset_info = DatasetInfo(dataset_name="test_dataset", format_version=hafnia.__dataset_format_version__)
    assert dataset_info.format_version == hafnia.__dataset_format_version__

    # Invalid version
    with pytest.raises(ValueError, match="Invalid format_version '.*'. Must be a valid version string"):
        DatasetInfo(dataset_name="test_dataset", format_version="invalid_version")


def test_dataset_format_version_is_newer_warning():
    from unittest.mock import patch

    c_format_version = Version(hafnia.__dataset_format_version__)
    n_format_version = f"{c_format_version.major}.{c_format_version.minor}.{c_format_version.micro + 1}"

    # Check warning is logged to the user logger. Because caplog/pytest.raises doesn't work with 'user_logger'
    with patch("hafnia.log.user_logger.warning") as mock_warning:
        DatasetInfo(dataset_name="test_dataset", format_version=n_format_version)
        mock_warning.assert_called_once()
        call_args = mock_warning.call_args[0][0]  # Get the first argument (message)
        assert "Please consider updating Hafnia package" in call_args
