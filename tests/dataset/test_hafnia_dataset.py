import os
from pathlib import Path

import pytest

from hafnia.dataset.dataset_names import ColumnName, DeploymentStage
from hafnia.dataset.dataset_upload_helper import dataset_info_from_dataset
from hafnia.dataset.hafnia_dataset import DatasetInfo, HafniaDataset, Sample, TaskInfo

# from data_management import utils
from hafnia.dataset.operations import dataset_stats, dataset_transformations
from hafnia.dataset.primitives.classification import Classification
from tests.helper_testing import (
    get_hafnia_functions_from_module,
    get_micro_hafnia_dataset,
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
            file_name=str(path),
            height=100,
            width=200,
            split="train",
            is_sample=False,
            classifications=[Classification(class_name="Class A", class_idx=0)],
        )
        for path in path_files
    ]
    dataset = HafniaDataset.from_samples_list(samples_list=samples, info=dataset_info)
    dataset.write(path_dataset)

    dataset_reloaded = HafniaDataset.from_path(path_dataset)
    assert dataset_reloaded.info == dataset.info
    table_expected = dataset.samples.drop(ColumnName.FILE_NAME)
    table_actual = dataset_reloaded.samples.drop(ColumnName.FILE_NAME)
    assert table_expected.equals(table_actual), "The samples tables do not match after reloading the dataset."


@pytest.mark.parametrize("function_name", get_hafnia_functions_from_module(dataset_transformations))
def test_hafnia_dataset_has_all_dataset_transforms(function_name: str):
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
    dataset_name = "tiny-dataset"
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


def test_dataset_stats():
    dataset = get_micro_hafnia_dataset(dataset_name="tiny-dataset", force_update=False)
    dataset.print_stats()
