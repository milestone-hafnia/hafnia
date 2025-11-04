from pathlib import Path
from typing import Callable

from hafnia.dataset import primitives
from hafnia.dataset.dataset_names import SampleField
from hafnia.dataset.format_conversions import format_yolo
from hafnia.dataset.hafnia_dataset import HafniaDataset, Sample
from tests.helper_testing import get_micro_hafnia_dataset, get_path_test_dataset_formats


def test_import_yolo_format_visualized(compare_to_expected_image: Callable) -> None:
    path_yolo_dataset = get_path_test_dataset_formats() / "format_yolo"

    hafnia_dataset = format_yolo.from_yolo_format(path_yolo_dataset)

    for sample_dict in hafnia_dataset:
        sample = Sample(**sample_dict)

    sample_visualized = sample.draw_annotations()
    compare_to_expected_image(sample_visualized)


def test_format_yolo_import_export_tiny_dataset(tmp_path: Path, compare_to_expected_image: Callable) -> None:
    dataset: HafniaDataset = get_micro_hafnia_dataset(dataset_name="micro-tiny-dataset")  # type: ignore[annotation-unchecked]

    path_yolo_dataset_exported = tmp_path / "exported_yolo_dataset"
    format_yolo.to_yolo_format(
        dataset=dataset,
        path_export_yolo_dataset=path_yolo_dataset_exported,
    )

    dataset_reloaded = format_yolo.from_yolo_format(path_yolo_dataset_exported)

    for sample_dict in dataset_reloaded:
        sample = Sample(**sample_dict)
        break

    sample_visualized = sample.draw_annotations()
    compare_to_expected_image(sample_visualized)


def test_format_yolo_import_export(tmp_path: Path) -> None:
    path_import_yolo_dataset = get_path_test_dataset_formats() / "format_yolo"
    path_expected_class_names = path_import_yolo_dataset / format_yolo.FILENAME_YOLO_CLASS_NAMES
    assert path_expected_class_names.exists()
    path_expected_images_txt = path_import_yolo_dataset / format_yolo.FILENAME_YOLO_IMAGES_TXT
    assert path_expected_images_txt.exists()

    # Test case 1: Import YOLO dataset
    dataset = format_yolo.from_yolo_format(path_import_yolo_dataset)

    assert len(dataset) == 3
    assert len(dataset.info.tasks) == 1
    task = dataset.info.tasks[0]
    assert len(task.class_names or []) == 80
    assert task.primitive == primitives.Bbox
    assert task.name == primitives.Bbox.default_task_name()

    # Test case 2: Export yolo dataset
    path_yolo_dataset_exported = tmp_path / "exported_yolo_dataset"
    format_yolo.to_yolo_format(
        dataset=dataset,
        path_export_yolo_dataset=path_yolo_dataset_exported,
        task_name=None,
    )
    path_class_names = path_yolo_dataset_exported / format_yolo.FILENAME_YOLO_CLASS_NAMES
    assert path_class_names.read_text() == path_expected_class_names.read_text()

    path_images_txt = path_yolo_dataset_exported / format_yolo.FILENAME_YOLO_IMAGES_TXT
    assert path_images_txt.read_text() == path_expected_images_txt.read_text().strip()

    for path_image in path_images_txt.read_text().splitlines():
        path_image_full = path_yolo_dataset_exported / path_image
        assert path_image_full.exists()
        path_label = path_image_full.with_suffix(".txt")
        assert path_label.exists()

    # Test case 3: Re-import exported YOLO dataset
    dataset_reimported = format_yolo.from_yolo_format(path_yolo_dataset_exported)

    assert len(dataset_reimported) == len(dataset)
    assert len(dataset_reimported.info.tasks) == len(dataset.info.tasks)
    actual_samples = dataset_reimported.samples.drop(SampleField.FILE_PATH)
    expected_samples = dataset.samples.drop(SampleField.FILE_PATH)
    assert actual_samples.equals(expected_samples)
