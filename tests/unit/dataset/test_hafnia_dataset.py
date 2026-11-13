import json
import os
from pathlib import Path

import pytest
from packaging.version import Version

import hafnia
from hafnia.dataset.dataset_names import FILENAME_ANNOTATIONS_JSONL, SampleField
from hafnia.dataset.hafnia_dataset import HafniaDataset

# from data_management import utils
from hafnia.dataset.hafnia_dataset_types import DatasetInfo, Sample, TaskInfo
from hafnia.dataset.operations import dataset_stats, dataset_transformations
from hafnia.dataset.primitives import PRIMITIVE_COLUMN_NAMES, Classification
from tests.helper_testing import (
    get_hafnia_functions_from_module,
    get_micro_hafnia_dataset,
)


def test_dataset_info_serializing_deserializing(tmp_path: Path):
    # Create a sample dataset
    tasks = [
        TaskInfo.from_class_names(name="Sample Task", class_names=["Class A", "Class B"], primitive=Classification),
        TaskInfo.from_class_names(name="Another Task", class_names=["Class C", "Class D"], primitive=Classification),
    ]
    dataset_info = DatasetInfo(dataset_name="Sample Dataset", version="1.0.0", tasks=tasks)

    path_dataset = tmp_path / "example_dataset_info.json"
    dataset_info.write_json(path_dataset)

    # Deserialize from JSON
    loaded_dataset_info = DatasetInfo.from_json_file(path_dataset)
    assert loaded_dataset_info == dataset_info


def test_hafnia_dataset_save_and_load(tmp_path: Path):
    # Create a sample dataset
    task_info = TaskInfo.from_class_names(
        name="Sample Task", class_names=["Class A", "Class B"], primitive=Classification
    )
    dataset_info = DatasetInfo(
        dataset_name="Sample Dataset", version="1.0.0", tasks=[task_info], primitive=Classification
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
    table_expected = dataset.samples.drop(SampleField.FILE_PATH)
    table_actual = dataset_reloaded.samples.drop(SampleField.FILE_PATH)
    assert table_expected.equals(table_actual), "The samples tables do not match after reloading the dataset."


def _write_sample_files(path_folder: Path, samples: list) -> list:
    path_folder.mkdir(parents=True, exist_ok=True)
    sample_paths = []
    for i, sample in enumerate(samples):
        path_sample = path_folder / f"sample{i}.jsonl"
        path_sample.write_text(sample.model_dump_json() + "\n")
        sample_paths.append(path_sample)
    return sample_paths


def _from_paths_test_setup(tmp_path: Path):
    task_info = TaskInfo.from_class_names(
        name="Sample Task", class_names=["Class A", "Class B"], primitive=Classification
    )
    dataset_info = DatasetInfo(dataset_name="Sample Dataset", version="1.0.0", tasks=[task_info])
    for name in ["image0.jpg", "image1.jpg", "override.jpg"]:
        (tmp_path / name).write_text("")
    samples = [
        Sample(
            file_path=name,  # Relative to the folder of the sample file
            height=100,
            width=200,
            split="train",
            classifications=[Classification(class_name="Class B", class_idx=1, task_name="Sample Task")],
        )
        for name in ["image0.jpg", "image1.jpg"]
    ]
    return dataset_info, samples


def test_from_paths(tmp_path: Path):
    dataset_info, samples = _from_paths_test_setup(tmp_path)
    sample_paths = _write_sample_files(tmp_path, samples)
    path_dataset_info = tmp_path / "dataset_info.json"
    dataset_info.write_json(path_dataset_info)

    media_paths = [None, tmp_path / "override.jpg"]
    dataset = HafniaDataset.from_paths(sample_paths, dataset_info=path_dataset_info, media_paths=media_paths)

    assert len(dataset) == 2
    assert dataset.info == dataset_info
    file_paths = dataset.samples[SampleField.FILE_PATH].to_list()
    assert file_paths == [(tmp_path / "image0.jpg").as_posix(), (tmp_path / "override.jpg").as_posix()]


def test_from_paths_raises_on_dataset_info_mismatch(tmp_path: Path):
    dataset_info, samples = _from_paths_test_setup(tmp_path)
    sample_paths = _write_sample_files(tmp_path, samples)

    dataset_info_other_task = dataset_info.model_copy(
        update={
            "tasks": [TaskInfo.from_class_names(name="Other Task", class_names=["Class B"], primitive=Classification)]
        }
    )
    with pytest.raises(ValueError, match="Other Task|Sample Task"):
        HafniaDataset.from_paths(sample_paths, dataset_info=dataset_info_other_task)

    dataset_info_other_classes = dataset_info.model_copy(
        update={
            "tasks": [TaskInfo.from_class_names(name="Sample Task", class_names=["Class A"], primitive=Classification)]
        }
    )
    with pytest.raises(ValueError, match="Class B"):
        HafniaDataset.from_paths(sample_paths, dataset_info=dataset_info_other_classes)

    dataset_info_other_order = dataset_info.model_copy(
        update={
            "tasks": [
                TaskInfo.from_class_names(
                    name="Sample Task", class_names=["Class B", "Class A"], primitive=Classification
                )
            ]
        }
    )
    with pytest.raises(ValueError, match="class index"):
        HafniaDataset.from_paths(sample_paths, dataset_info=dataset_info_other_order)


def test_from_paths_allows_tasks_without_annotations(tmp_path: Path):
    dataset_info, samples = _from_paths_test_setup(tmp_path)
    sample_paths = _write_sample_files(tmp_path, samples)
    task_without_annotations = TaskInfo.from_class_names(
        name="Unused Task", class_names=["Class C"], primitive=Classification
    )
    dataset_info = dataset_info.model_copy(update={"tasks": dataset_info.tasks + [task_without_annotations]})

    dataset = HafniaDataset.from_paths(sample_paths, dataset_info=dataset_info)

    dataset.check_dataset_tasks(require_annotations_for_all_tasks=False)
    with pytest.raises(ValueError, match="has no task.name='Unused Task' objects"):
        dataset.check_dataset_tasks()


def test_from_paths_validation_errors(tmp_path: Path):
    dataset_info, samples = _from_paths_test_setup(tmp_path)
    sample_paths = _write_sample_files(tmp_path, samples)

    with pytest.raises(ValueError, match="same length"):
        HafniaDataset.from_paths(sample_paths, dataset_info=dataset_info, media_paths=[None])

    with pytest.raises(FileNotFoundError):
        HafniaDataset.from_paths(sample_paths, dataset_info=dataset_info, media_paths=[None, tmp_path / "missing.jpg"])

    path_empty = tmp_path / "empty.jsonl"
    path_empty.write_text("\n")
    with pytest.raises(ValueError, match="No samples found"):
        HafniaDataset.from_paths([path_empty], dataset_info=dataset_info)

    path_unsupported = tmp_path / "sample.txt"
    path_unsupported.write_text(samples[0].model_dump_json())
    with pytest.raises(ValueError, match="Unsupported file format"):
        HafniaDataset.from_paths([path_unsupported], dataset_info=dataset_info)

    path_invalid = tmp_path / "invalid.json"
    path_invalid.write_text('{"height": 100}')
    with pytest.raises(ValueError, match="invalid.json"):
        HafniaDataset.from_paths([path_invalid], dataset_info=dataset_info)


def test_from_paths_read_errors(tmp_path: Path):
    dataset_info, samples = _from_paths_test_setup(tmp_path)

    # Invalid line in a '.jsonl' file is reported with file and line number
    path_jsonl = tmp_path / "bad_line.jsonl"
    path_jsonl.write_text(samples[0].model_dump_json() + "\n\n" + '{"height": 100}\n')
    with pytest.raises(ValueError, match="line 3 of sample file '.*bad_line.jsonl'"):
        HafniaDataset.from_paths([path_jsonl], dataset_info=dataset_info)

    # A binary file (e.g. an image passed by mistake) gives a 'ValueError' naming the file
    path_binary = tmp_path / "binary.jsonl"
    path_binary.write_bytes(bytes([0xFF, 0xD8, 0xFF, 0xE0]) + b"\n")
    with pytest.raises(ValueError, match="binary.jsonl"):
        HafniaDataset.from_paths([path_binary], dataset_info=dataset_info)

    # Unsupported formats are rejected before any file is read
    path_image = tmp_path / "image0.jpg"
    with pytest.raises(ValueError, match="Unsupported file format '.jpg'"):
        Sample.samples_from_files([path_jsonl, path_image])

    # A sample without 'file_path' and no media file override is rejected when checking for images
    sample_without_path = samples[0].model_copy(update={"file_path": None})
    sample_paths = _write_sample_files(tmp_path / "no_path", [sample_without_path])
    with pytest.raises(ValueError, match="1 samples have no 'file_path'"):
        HafniaDataset.from_paths(sample_paths, dataset_info=dataset_info)


def test_samples_from_files(tmp_path: Path):
    _, samples = _from_paths_test_setup(tmp_path)
    sample_paths = _write_sample_files(tmp_path, samples)

    assert len(list(Sample.samples_from_file(sample_paths[0]))) == 1

    samples_iterator = Sample.samples_from_files(sample_paths, media_paths=[tmp_path / "override.jpg", None])
    file_paths = [sample.file_path for sample in samples_iterator]
    assert file_paths == [(tmp_path / "override.jpg").as_posix(), (tmp_path / "image1.jpg").as_posix()]

    with pytest.raises(FileNotFoundError):
        Sample.samples_from_files(sample_paths + [tmp_path / "missing.jsonl"])


def test_from_paths_multiple_samples_per_file(tmp_path: Path):
    dataset_info, samples = _from_paths_test_setup(tmp_path)

    # JSONL with multiple samples, including a duplicate (e.g. independent predictions for the same image)
    path_jsonl = tmp_path / "multi.jsonl"
    path_jsonl.write_text("\n".join(sample.model_dump_json() for sample in [samples[0], samples[1], samples[1]]))
    # JSON with a list of samples and JSON with a single sample object
    path_json_list = tmp_path / "list.json"
    path_json_list.write_text(json.dumps([sample.model_dump(mode="json") for sample in samples]))
    path_json_object = tmp_path / "object.json"
    path_json_object.write_text(samples[0].model_dump_json())

    media_paths = [None, tmp_path / "override.jpg", None]
    dataset = HafniaDataset.from_paths(
        [path_jsonl, path_json_list, path_json_object], dataset_info=dataset_info, media_paths=media_paths
    )

    path_image0, path_image1, path_override = [
        (tmp_path / n).as_posix() for n in ["image0.jpg", "image1.jpg", "override.jpg"]
    ]
    expected_file_paths = [path_image0, path_image1, path_image1, path_override, path_override, path_image0]
    assert dataset.samples[SampleField.FILE_PATH].to_list() == expected_file_paths
    assert dataset.samples[SampleField.SAMPLE_INDEX].to_list() == list(range(6))


def test_write_annotations_stores_posix_paths(tmp_path: Path):
    task_info = TaskInfo.from_class_names(
        name="Sample Task", class_names=["Class A", "Class B"], primitive=Classification
    )
    dataset_info = DatasetInfo(dataset_name="Sample Dataset", version="1.0.0", tasks=[task_info])
    path_dataset = tmp_path / "test_hafnia_dataset"
    path_file = path_dataset / "data" / "img.png"
    path_file.parent.mkdir(parents=True, exist_ok=True)
    path_file.write_text("")

    sample = Sample(
        file_path=str(path_file),
        height=10,
        width=10,
        split="train",
        classifications=[Classification(class_name="Class A", class_idx=0)],
    )
    dataset = HafniaDataset.from_samples_list(samples_list=[sample], info=dataset_info)
    dataset.write_annotations(path_folder=path_dataset)

    with open(path_dataset / FILENAME_ANNOTATIONS_JSONL) as f:
        stored_path = json.loads(f.readline())[SampleField.FILE_PATH]
    assert "\\" not in stored_path, f"Backslash found in stored path: {stored_path!r}"


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


def test_task_info_validation_exceptions():
    with pytest.raises(ValueError, match="Class names must be unique"):
        TaskInfo.from_class_names(
            primitive=Classification,
            class_names=["car", "person", "car"],  # <-- Duplicate name is used
        )
    primitive_wrong_name = "WrongPrimitiveName"
    with pytest.raises(ValueError, match=f"Primitive '{primitive_wrong_name}' is not recognized."):
        TaskInfo.from_class_names(
            primitive=primitive_wrong_name,
            class_names=["person", "car"],  # <-- Duplicate name is used
        )


def test_dataset_info_validation_exceptions():
    # Use case 1: Same primitive - different task name is allowed!
    DatasetInfo(
        dataset_name="test_dataset",
        version="1.0.0",
        tasks=[
            TaskInfo.from_class_names(primitive=Classification, class_names=["car", "person"]),
            TaskInfo.from_class_names(primitive=Classification, class_names=["car", "person"], name="Task2"),
        ],
    )

    # Use case 2: Same primitive and same task name is NOT allowed
    with pytest.raises(ValueError, match="Tasks must be unique"):
        DatasetInfo(
            dataset_name="test_dataset",
            version="1.0.0",
            tasks=[
                TaskInfo.from_class_names(primitive=Classification, class_names=["car", "person"], name="my_task"),
                TaskInfo.from_class_names(primitive=Classification, class_names=["car", "person"], name="my_task"),
            ],
        )


def test_dataset_info_replace_task():
    task1 = TaskInfo.from_class_names(primitive=Classification, class_names=["car", "person"], name="Task1")
    task2 = TaskInfo.from_class_names(primitive=Classification, class_names=["cat", "dog"], name="Task2")
    dataset_info = DatasetInfo(
        dataset_name="test_dataset",
        version="1.0.0",
        tasks=[task1, task2],
    )

    # Create a new task to replace task1
    new_task1 = TaskInfo.from_class_names(primitive=Classification, class_names=["bus", "truck"], name="Task3")

    # Replace task1 with new_task1
    dataset_info_updated = dataset_info.replace_task(old_task=task1, new_task=new_task1)

    # Verify that the task has been updated
    assert len(dataset_info_updated.tasks) == 2
    assert new_task1 in dataset_info_updated.tasks
    assert task2 in dataset_info_updated.tasks
    assert task1 not in dataset_info_updated.tasks

    # Attempt to replace a non-existing task
    non_existing_task = TaskInfo.from_class_names(
        primitive=Classification, class_names=["bike"], name="NonExistingTask"
    )
    with pytest.raises(ValueError, match="Task '.*' not found in dataset info."):
        dataset_info.replace_task(old_task=non_existing_task, new_task=new_task1)


def test_dataset_version_validation():
    # Valid version
    DatasetInfo(dataset_name="test_dataset", version="1.0.0", tasks=[])

    # Invalid version
    with pytest.raises(ValueError, match="Invalid version string"):
        DatasetInfo(dataset_name="test_dataset", version="invalid_version")


def test_dataset_format_version_validation():
    # Valid dataset format version
    dataset_info = DatasetInfo(dataset_name="test_dataset", tasks=[])
    assert dataset_info.format_version == hafnia.__dataset_format_version__

    # Explicitly set valid version
    dataset_info = DatasetInfo(dataset_name="test_dataset", format_version=hafnia.__dataset_format_version__, tasks=[])
    assert dataset_info.format_version == hafnia.__dataset_format_version__

    # Invalid version
    with pytest.raises(ValueError, match="Invalid version string"):
        DatasetInfo(dataset_name="test_dataset", format_version="invalid_version", tasks=[])


def test_dataset_format_version_is_newer_warning():
    from unittest.mock import patch

    c_format_version = Version(hafnia.__dataset_format_version__)
    n_format_version = f"{c_format_version.major}.{c_format_version.minor}.{c_format_version.micro + 1}"

    # Check warning is logged to the user logger. Because caplog/pytest.raises doesn't work with 'user_logger'
    with patch("hafnia.log.user_logger.warning") as mock_warning:
        DatasetInfo(dataset_name="test_dataset", format_version=n_format_version, tasks=[])
        mock_warning.assert_called_once()
        call_args = mock_warning.call_args[0][0]  # Get the first argument (message)
        assert "Please consider updating Hafnia package" in call_args


def test_define_sample_set_by_size_validation_exceptions():
    dataset = get_micro_hafnia_dataset(dataset_name="micro-tiny-dataset")
    n_dataset_samples = len(dataset)

    # Use case 1: A negative number of samples is not allowed
    with pytest.raises(ValueError, match="must be a positive number of samples"):
        dataset.define_sample_set_by_size(n_samples=-1)

    # Use case 2: More samples than the dataset contains is not allowed
    with pytest.raises(ValueError, match="exceeds the number of samples in the dataset"):
        dataset.define_sample_set_by_size(n_samples=n_dataset_samples + 1)

    # Use case 3: Tagging all samples of the dataset is allowed
    dataset_all_tagged = dataset.define_sample_set_by_size(n_samples=n_dataset_samples)
    assert len(dataset_all_tagged.create_sample_dataset()) == n_dataset_samples


def test_without_annotations(tmp_path: Path):
    dataset = get_micro_hafnia_dataset(dataset_name="micro-tiny-dataset")
    label_columns = [name for name in [*PRIMITIVE_COLUMN_NAMES, SampleField.TAGS] if name in dataset.samples.columns]
    assert all(dataset.samples[name].list.len().sum() > 0 for name in label_columns)

    label_free = dataset.without_annotations()

    assert all(label_free.samples[name].list.len().sum() == 0 for name in label_columns)
    assert label_free.samples[SampleField.META].null_count() == len(label_free)
    assert label_free.samples.schema == dataset.samples.schema
    assert label_free.info == dataset.info
    assert label_free.samples[SampleField.SAMPLE_INDEX].equals(dataset.samples[SampleField.SAMPLE_INDEX])
    assert dataset.samples[SampleField.BBOXES].list.len().sum() > 0  # The input is not modified

    label_free.update_samples(label_free.samples.drop(SampleField.FILE_PATH)).write_annotations(path_folder=tmp_path)
    reloaded = HafniaDataset.from_path(tmp_path, check_for_images=False)
    assert reloaded.samples[SampleField.META].dtype == dataset.samples[SampleField.META].dtype
