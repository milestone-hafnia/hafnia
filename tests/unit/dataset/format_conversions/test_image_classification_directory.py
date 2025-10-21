from pathlib import Path

import pytest

from hafnia.dataset.format_conversions.image_classification_from_directory import (
    export_image_classification_directory_tree,
    import_image_classification_directory_tree,
)
from hafnia.dataset.hafnia_dataset import HafniaDataset
from tests.helper_testing import get_micro_hafnia_dataset


def test_import_export_image_classification_from_directory(tmp_path: Path) -> None:
    dataset: HafniaDataset = get_micro_hafnia_dataset(dataset_name="micro-tiny-dataset")

    path_exported = tmp_path / "exported"
    with pytest.raises(ValueError, match="Found multiple tasks"):
        export_image_classification_directory_tree(dataset, path_output=path_exported)

    task = dataset.info.get_task_by_task_name_and_primitive(task_name="Time of Day", primitive=None)
    path_dataset_exported = export_image_classification_directory_tree(
        dataset, path_output=path_exported, task_name=task.name, clean_folder=True
    )

    actual_class_names = [p.name for p in path_dataset_exported.iterdir()]
    assert len(actual_class_names) == 3
    expected_class_names = [n.replace("/", "_") for n in task.class_names or []]
    assert set(actual_class_names).issubset(set(expected_class_names))
    hafnia_dataset_imported = import_image_classification_directory_tree(
        path_folder=path_dataset_exported,
        split="train",
        n_samples=None,
    )

    assert len(hafnia_dataset_imported.info.tasks) == 1
    assert hafnia_dataset_imported.info.tasks[0].primitive == task.primitive
    assert len(hafnia_dataset_imported.samples) == len(dataset.samples)

    hafnia_dataset_imported = import_image_classification_directory_tree(
        path_folder=path_dataset_exported,
        split="train",
        n_samples=2,
    )

    assert len(hafnia_dataset_imported.info.tasks) == 1
    assert hafnia_dataset_imported.info.tasks[0].primitive == task.primitive
    assert len(hafnia_dataset_imported.samples) == 2
