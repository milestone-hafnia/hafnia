from datetime import datetime
from pathlib import Path

import pytest

from hafnia.dataset.dataset_details_uploader import dataset_details_from_hafnia_dataset
from hafnia.dataset.dataset_names import SampleField
from hafnia.dataset.hafnia_dataset import HafniaDataset
from hafnia.dataset.primitives.classification import Classification
from tests import helper_testing


@pytest.mark.parametrize("dataset_name", helper_testing.MICRO_DATASETS)
def test_dataset_details_from_hafnia_dataset(dataset_name: str, tmp_path: Path):
    path_dataset = helper_testing.get_path_micro_hafnia_dataset(dataset_name=dataset_name, force_update=False)
    dataset = HafniaDataset.from_path(path_dataset)
    classification_tasks = dataset.info.get_tasks_by_primitive(Classification)
    distribution_task_names = [task.name or task.primitive.default_task_name() for task in classification_tasks]
    gallery_image_names = [dataset.samples[SampleField.FILE_PATH].str.split("/").list.last().sort()[0]]
    dataset_info = dataset_details_from_hafnia_dataset(
        dataset=dataset,
        path_gallery_images=tmp_path / "gallery_images",
        gallery_image_names=gallery_image_names,
        distribution_task_names=distribution_task_names,
    )
    # Check if dataset info can be serialized to JSON
    dataset_info_json = dataset_info.model_dump_json()  # noqa: F841

    # Basic checks on dataset info fields
    assert isinstance(dataset_info.dataset_updated_at, datetime)
    assert isinstance(dataset_info.version, str) and len(dataset_info.version) > 0
    assert isinstance(dataset_info.dataset_format_version, str) and len(dataset_info.dataset_format_version) > 0

    # Check dataset variants
    assert dataset_info.dataset_variants is not None
    assert len(dataset_info.dataset_variants) > 0

    # Check split annotations reports
    assert dataset_info.split_annotations_reports is not None
    assert len(dataset_info.split_annotations_reports) == 8
    full_split_annotations_report = [r for r in dataset_info.split_annotations_reports if r.split == "full"]
    assert len(full_split_annotations_report) == 2, "There should be exactly two 'full' split annotations report"
    full_report = full_split_annotations_report[0]

    # Check annotated object reports
    assert full_report.annotated_object_reports is not None
    assert len(full_report.annotated_object_reports) > 0
    expected_primitives = {t.primitive.__name__ for t in dataset.info.tasks}
    actual_primitives = {r.obj.annotation_type.name for r in full_report.annotated_object_reports}
    assert expected_primitives == actual_primitives

    # Check distribution values
    assert full_report.distribution_values is not None
    actual_dist_names = {d.distribution_category.distribution_type.name for d in full_report.distribution_values}
    assert actual_dist_names == set(distribution_task_names)
