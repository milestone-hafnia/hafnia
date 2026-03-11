from pathlib import Path
from typing import Dict, List

import pytest

from hafnia import utils
from hafnia.dataset import primitives
from hafnia.dataset.format_conversions.format_encord import FILENAME_ENCORD_ANNOTATIONS
from hafnia.dataset.hafnia_dataset import HafniaDataset
from hafnia.dataset.hafnia_dataset_types import ClassInfo, Sample, TaskInfo
from tests import helper_testing


def get_encord_dataset_compressed_path() -> Path:
    path_encord_dataset_folder = helper_testing.get_path_test_dataset_formats() / "format_encord"
    path_compressed_data = path_encord_dataset_folder / FILENAME_ENCORD_ANNOTATIONS
    return path_compressed_data


def get_encord_dataset() -> Dict[str, Dict]:
    path_compressed_data = get_encord_dataset_compressed_path()
    metadata_dict = utils.read_compressed_json(path_compressed_data)
    return metadata_dict


def test_parse_nested_ontology():
    metadata_dict = get_encord_dataset()

    ontology_dict = metadata_dict["ontology"]
    tasks = TaskInfo.from_encord_ontology_dict(ontology_dict)
    object_tasks = [task for task in tasks if task.primitive == primitives.Bbox]
    assert len(object_tasks) == 1, "Only 1 bbox task is expected as same primitive tasks are merged into one TaskInfo"
    object_task: TaskInfo = object_tasks[0]  # type: ignore
    assert len(object_task.classes) > 5, "Expected to find multiple classes for the object task"
    assert object_task.name == primitives.Bbox.default_task_name(), "Expected the object task to have a name"

    # Check that Vehicle class has expected attributes (Vehicle Color, Vehicle Type, Vehicle State)
    vehicle_classes = [cls for cls in object_task.classes if cls.name == "Vehicle"]
    assert len(vehicle_classes) == 1, "Expected to find one Vehicle class in the object task"
    vehicle_class: ClassInfo = vehicle_classes[0]  # type: ignore
    vehicle_attributes = {attr.name for attr in vehicle_class.attributes}
    assert {"Vehicle Color", "Vehicle Type", "Vehicle State"}.issubset(vehicle_attributes), (
        "Expected Vehicle class to have attributes"
    )

    # Check that Vehicle Color attribute has colors (Red, Blue, Green, Gray) and nested colors (Gray.Dark, Gray.Light)
    vehicle_color_task = [attr for attr in vehicle_class.attributes if attr.name == "Vehicle Color"]
    assert len(vehicle_color_task) == 1, "Expected to find Vehicle Color attribute for Vehicle class"
    vehicle_color_task: TaskInfo = vehicle_color_task[0]  # type: ignore
    assert {"Red", "Blue", "Gray"}.issubset(vehicle_color_task.get_class_names()), "Expected at least Red, Blue, Gray"
    gray_classes = [option for option in vehicle_color_task.classes if option.name == "Gray"]
    assert len(gray_classes) == 1, "Expected to find nested 'Gray' option for Vehicle Color"
    gray_class: ClassInfo = gray_classes[0]  # type: ignore

    # Check that Gray has nested options like Gray.Dark, Gray.Light
    assert len(gray_class.attributes) == 1, "Expected Gray to have nested attributes"
    gray_task = gray_class.attributes[0]  # type: ignore
    assert gray_task.primitive == primitives.Classification, (
        "Expected nested attribute of Gray to be a Classification task"
    )
    assert len(gray_task.classes) > 1, "Expected to find multiple nested attributes for Gray"
    class_names = set(gray_task.get_class_names())
    assert {"light", "medium"}.issubset(class_names), (
        "Expected to find Gray.Light and Gray.Medium nested options for Gray"
    )

    classification_tasks = [t for t in tasks if t.primitive.__name__ == "Classification"]
    assert len(classification_tasks) > 1, "Expected to find multiple classification tasks"


@pytest.fixture(scope="session")
def encord_dataset():  # This is session scoped fixture. Use 'dataset = dataset.copy()' for each test to avoid affecting other tests when modifying the dataset.
    path_compressed_data = get_encord_dataset_compressed_path()
    dataset = HafniaDataset.from_encord_zip_format(path_compressed_data)
    return dataset


def test_from_encord_zip_format(encord_dataset: HafniaDataset):
    encord_dataset = encord_dataset.copy()  # This is Session scoped fixture - copy to not affect other tests

    assert len(encord_dataset.info.tasks) > 0, "Expected to find tasks in the dataset info"
    assert len(encord_dataset.samples) > 0, "Expected to find samples in the dataset"

    # Check pydantic parsing of sample
    sample = Sample(**encord_dataset[0])
    assert len(sample.polygons or []) > 0, "Expected to find polygons in the sample"
    assert len(sample.bboxes or []) > 0, "Expected to find bboxes in the sample"
    for bbox in sample.bboxes or []:
        assert len(bbox.classifications or []) > 0, "Expected attributes for the bbox annotation"
    assert len(sample.bitmasks or []) > 0, "Expected to find bitmasks in the sample"
    assert len(sample.classifications or []) > 0, "Expected to find classifications in the sample"

    encord_dataset.check_dataset(check_splits=False)


def test_flattening_specification_assert(encord_dataset: HafniaDataset):
    encord_dataset = encord_dataset.copy()  # This is Session scoped fixture - copy to not affect other tests

    flattening_specification: Dict[TaskInfo, List[List[str]]] = {
        # Deliberately use wrong task names to trigger the assertion error
        TaskInfo(primitive=primitives.Classification, name="Nonexistent Task"): [
            ["Sunrise/Sunset Type"],
            ["Twilight Type"],
        ],
    }

    with pytest.raises(ValueError, match="does not exist in the dataset"):
        encord_dataset.flattening_by_specification(flattening_specification=flattening_specification)


def test_flattening_specification(encord_dataset: HafniaDataset):
    encord_dataset = encord_dataset.copy()  # This is Session scoped fixture - copy to not affect other tests

    # Check dataset looks correct
    assert len(encord_dataset.info.tasks) > 0, "Expected to find tasks in the dataset info"
    assert len(encord_dataset.samples) > 0, "Expected to find samples in the dataset"

    # Check classes names before flattening
    class_name_counts = encord_dataset.calculate_class_counts()
    class_names = [cn["Class Name"] for cn in class_name_counts]

    assert all("." not in cn for cn in class_names), "Expected not nesting and therefore no '.' before flattening"
    flattening_specification: Dict[TaskInfo, List[List[str]]] = {
        # Vehicle color is used to check nested attributes, but "Vehicle Type" would be a more natural choice.
        TaskInfo(primitive=primitives.Bbox): [["Vehicle Color", "Gray Tone"], ["Annotator Marking Type"]],
        TaskInfo(primitive=primitives.Polygon): [["Annotator Marking Polygon Type"]],
        TaskInfo(primitive=primitives.Bitmask): [["Annotator Marking Bitmask Type"]],
        TaskInfo(primitive=primitives.Classification, name="Time of Day"): [["Sunrise/Sunset Type"], ["Twilight Type"]],
    }

    dataset_flat = encord_dataset.flattening_by_specification(
        flattening_specification=flattening_specification,
    )

    # Check that class names are flattened as expected
    class_name_counts = dataset_flat.calculate_class_counts()
    class_names = [cn["Class Name"] for cn in class_name_counts]

    assert len([cn for cn in class_names if cn == "Vehicle"]) == 1, "Expected base class after flattening"
    assert len([cn for cn in class_names if cn == "Vehicle.Gray"]) == 1, "Expected base class after flattening"
    assert len([cn for cn in class_names if cn == "Sunrise/Sunset"]) == 1, "Expected base class after flattening"
    nested_names = [cn for cn in class_names if "." in cn]

    assert len([cn for cn in nested_names if cn.startswith("Vehicle.")]) > 8
    assert len([cn for cn in nested_names if cn.startswith("Vehicle.Gray.")]) > 0
    assert len([cn for cn in nested_names if cn.startswith("Annotator Marking.")]) > 0
    assert len([cn for cn in nested_names if cn.startswith("Annotator Marking Bitmask.")]) > 0
    assert len([cn for cn in nested_names if cn.startswith("Annotator Marking Polygon.")]) > 0
    assert len([cn for cn in nested_names if cn.startswith("Sunrise/Sunset.")]) > 0
    assert len([cn for cn in nested_names if cn.startswith("Twilight.")]) > 0

    # Check task info are updated as expected
    for task_info in dataset_flat.info.tasks:
        if task_info.primitive == primitives.Bbox:
            class_info = task_info.get_class_by_name("Vehicle.Red")
            assert class_info is not None, "Expected to find Vehicle.Red class after flattening"
            assert len(class_info.attributes or []) > 0, "Expected Vehicle.Red to have attributes after flattening"
            assert any(cls.name.startswith("Vehicle.Gray.") for cls in task_info.classes or []), (
                "Expected to find flattened Vehicle classes in the flattened dataset"
            )
        elif task_info.primitive == primitives.Polygon:
            assert any(cls.name.startswith("Annotator Marking Polygon.") for cls in task_info.classes or []), (
                "Expected to find flattened Annotator Marking Polygon classes in the flattened dataset"
            )
        elif task_info.primitive == primitives.Bitmask:
            assert any(cls.name.startswith("Annotator Marking Bitmask.") for cls in task_info.classes or []), (
                "Expected to find flattened Annotator Marking Bitmask classes in the flattened dataset"
            )
        elif task_info.primitive == primitives.Classification and task_info.name == "Time of Day":
            assert any(cls.name.startswith("Sunrise/Sunset.") for cls in task_info.classes or []), (
                "Expected to find flattened Sunrise/Sunset classes in the flattened dataset"
            )
            assert any(cls.name.startswith("Twilight.") for cls in task_info.classes or []), (
                "Expected to find flattened Twilight classes in the flattened dataset"
            )

    encord_dataset.check_dataset(check_splits=False)
