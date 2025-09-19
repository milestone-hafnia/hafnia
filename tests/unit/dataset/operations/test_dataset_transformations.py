import pytest

from hafnia.dataset.dataset_names import FieldName
from hafnia.dataset.hafnia_dataset import HafniaDataset, TaskInfo
from hafnia.dataset.operations.dataset_transformations import (
    get_task_info_from_task_name_and_primitive,
)
from hafnia.dataset.primitives import Bbox, Classification
from tests.helper_testing import get_path_micro_hafnia_dataset, get_strict_class_mapping_midwest


def test_class_mapper_strict():
    dataset_name = "tiny-dataset"
    path_dataset = get_path_micro_hafnia_dataset(dataset_name=dataset_name, force_update=False)
    dataset = HafniaDataset.from_path(path_dataset)

    dataset_updated = dataset.class_mapper_strict(
        strict_class_mapping=get_strict_class_mapping_midwest(),
        primitive=Bbox,
    )

    dataset_updated.check_dataset_tasks()

    task_bbox = [t for t in dataset_updated.info.tasks if t.primitive == Bbox][0]
    expected_class_names = ["person", "vehicle", "truck"]
    expected_indices = list(range(len(expected_class_names)))
    assert task_bbox.class_names == expected_class_names
    bboxes = dataset_updated.samples[task_bbox.primitive.column_name()].explode().struct.unnest()
    assert set(bboxes[FieldName.CLASS_NAME]).issubset(expected_class_names)
    assert set(bboxes[FieldName.CLASS_IDX]).issubset(set(expected_indices))
    assert not dataset.samples.equals(dataset_updated.samples), (
        "Samples should be different after class mapping. Verify that original 'dataset.samples' is not mutated."
    )
    assert dataset.info != dataset_updated.info, (
        "Info should be different after class mapping. Verify that original 'dataset.info' is not mutated."
    )


def test_rename_task():
    dataset_name = "tiny-dataset"
    path_dataset = get_path_micro_hafnia_dataset(dataset_name=dataset_name, force_update=False)
    dataset = HafniaDataset.from_path(path_dataset)

    old_task_name = Bbox.default_task_name()
    new_task_name = "renamed_objects"
    dataset_renamed = dataset.rename_task(old_task_name=old_task_name, new_task_name=new_task_name)

    dataset_renamed.check_dataset_tasks()

    task_bbox = [t for t in dataset_renamed.info.tasks if t.primitive == Bbox][0]
    assert task_bbox.name == new_task_name
    bboxes = dataset_renamed.samples[task_bbox.primitive.column_name()].explode().struct.unnest()
    assert FieldName.TASK_NAME in bboxes.columns
    assert set(bboxes[FieldName.TASK_NAME]) == {new_task_name}
    assert not dataset.samples.equals(dataset_renamed.samples), (
        "Samples should be different after renaming task. Verify that original 'dataset.samples' is not mutated."
    )
    assert dataset.info != dataset_renamed.info, (
        "Info should be different after renaming task. Verify that original 'dataset.info' is not mutated."
    )


def test_merge_datasets():
    dataset_name = "tiny-dataset"
    path_dataset = get_path_micro_hafnia_dataset(dataset_name=dataset_name, force_update=False)
    dataset = HafniaDataset.from_path(path_dataset)

    dataset_0 = dataset.select_samples(n_samples=2, seed=42)
    dataset_1 = dataset.select_samples(n_samples=2, seed=43)

    # Use case 1: Merging two datasets with the same tasks and class names
    dataset_merged = HafniaDataset.from_merge(dataset0=dataset_0, dataset1=dataset_1)
    assert len(dataset_merged.samples) == 4, f"Expected 4 samples, got {len(dataset_merged.samples)}"
    assert len(dataset_merged.info.tasks) == len(dataset_0.info.tasks), "Tasks should be preserved after merging"
    assert len(dataset_merged.info.tasks) == len(dataset_1.info.tasks), "Tasks should be preserved after merging"

    # Use case 2: Merging two datasets with the same tasks but different class names should raise an error
    mapping = get_strict_class_mapping_midwest()
    dataset_1_changed = dataset_1.class_mapper_strict(
        strict_class_mapping=mapping,
        primitive=Bbox,
    )
    with pytest.raises(ValueError, match="Cannot merge datasets with different class names for the same task name"):
        HafniaDataset.from_merge(dataset0=dataset_0, dataset1=dataset_1_changed)


def test_select_samples_by_class_name():
    dataset_name = "tiny-dataset"
    path_dataset = get_path_micro_hafnia_dataset(dataset_name=dataset_name, force_update=False)
    dataset = HafniaDataset.from_path(path_dataset)

    # Use case 1: Select samples by class name (task_name and primitive are auto-inferred)
    dataset_updated = dataset.select_samples_by_class_name(name="Vehicle.Car", primitive=Bbox)
    assert len(dataset_updated) < len(dataset), "Expected fewer samples after filtering by class name"

    # Use case 2: Select samples by class name with explicit 'primitive'
    dataset_updated = dataset.select_samples_by_class_name(name="Vehicle.Car", primitive=Bbox)
    assert len(dataset_updated) < len(dataset), "Expected fewer samples after filtering by class name"

    # Use case 3: Wrong class name (not found in any task)
    with pytest.raises(ValueError, match="The specified names"):
        dataset_updated = dataset.select_samples_by_class_name(name="NonExistingClass", primitive=Bbox)

    # Use case 4: Class exists but not in the specified task
    with pytest.raises(ValueError, match="The specified names"):
        dataset_updated = dataset.select_samples_by_class_name(name="Vehicle.Car", task_name="Weather")


def test_get_task_info_from_task_name_and_primitive():
    task_class = TaskInfo(primitive="Classification", class_names=["cat", "dog"])
    task_bbox = TaskInfo(primitive="Bbox", class_names=["car", "bus"])

    ### Test case: No tasks defined
    with pytest.raises(ValueError, match="Dataset has no tasks defined."):
        get_task_info_from_task_name_and_primitive(tasks=[])

    ### Test case: Task name and primitive is not required when there is only one task
    get_task_info_from_task_name_and_primitive(tasks=[task_class])

    ### Test case: Task name is required when there are multiple tasks
    with pytest.raises(ValueError, match="For multiple tasks"):
        get_task_info_from_task_name_and_primitive(tasks=[task_class, task_bbox])

    ### Test case: Task name is found
    task_actual = get_task_info_from_task_name_and_primitive(tasks=[task_class, task_bbox], task_name=task_class.name)
    assert task_actual == task_class

    ### Test case: Task name is not found
    with pytest.raises(ValueError, match="No task found with task_name="):
        get_task_info_from_task_name_and_primitive(tasks=[task_class, task_bbox], task_name="segmentation")

    ### Test case: Non-unique task name
    task_bbox_bad = task_bbox.model_copy()
    task_bbox_bad.name = task_class.name
    with pytest.raises(ValueError, match="Found multiple tasks with task_name="):
        get_task_info_from_task_name_and_primitive(tasks=[task_class, task_bbox_bad], task_name=task_class.name)

    ### Test case: Primitive is found by string
    task_actual = get_task_info_from_task_name_and_primitive(tasks=[task_class, task_bbox], primitive="Classification")
    assert task_actual == task_class

    ### Test case: Primitive is found by type
    task_actual = get_task_info_from_task_name_and_primitive(tasks=[task_class, task_bbox], primitive=Classification)
    assert task_actual == task_class

    ### Test case: Primitive is not found
    with pytest.raises(ValueError, match="No task found with primitive="):
        get_task_info_from_task_name_and_primitive(tasks=[task_class, task_bbox], primitive="Polygon")

    ### Test case: Non-unique Primitive
    task_bbox_bad = task_bbox.model_copy()
    task_bbox_bad.primitive = task_class.primitive
    with pytest.raises(ValueError, match="Found multiple tasks with primitive="):
        get_task_info_from_task_name_and_primitive(tasks=[task_class, task_bbox_bad], primitive=task_class.primitive)

    ### Test case: Both task name and primitive are provided and match
    task_actual = get_task_info_from_task_name_and_primitive(
        tasks=[task_class, task_bbox], task_name=task_class.name, primitive=task_class.primitive
    )
    assert task_actual == task_class

    ### Test case: Both task name and primitive are provided but do not match
    with pytest.raises(ValueError, match="No task found with task_name="):
        get_task_info_from_task_name_and_primitive(
            tasks=[task_class, task_bbox], task_name=task_class.name, primitive=task_bbox.primitive
        )

    ### Test case: Both task name and primitive are provided but do not match
    with pytest.raises(ValueError, match="This should never happen"):
        get_task_info_from_task_name_and_primitive(
            tasks=[task_class, task_class], task_name=task_class.name, primitive=task_class.primitive
        )
