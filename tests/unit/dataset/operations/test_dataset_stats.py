import pytest

import hafnia
from hafnia.dataset.hafnia_dataset import HafniaDataset
from hafnia.dataset.primitives.bbox import Bbox
from tests.helper_testing import (
    MICRO_DATASETS,
    get_micro_hafnia_dataset,
    get_path_micro_hafnia_dataset,
    get_path_workspace,
)


@pytest.mark.parametrize("micro_dataset_name", MICRO_DATASETS)
def test_micro_dataset_format_versions(micro_dataset_name: str):
    force_update = False  # Use this flag to update the micro test datasets if needed
    path_dataset = get_path_micro_hafnia_dataset(dataset_name=micro_dataset_name, force_update=force_update)
    path_dataset_relative = path_dataset.relative_to(get_path_workspace())
    dataset = HafniaDataset.from_path(path_dataset)
    format_version_match = dataset.info.format_version == hafnia.__dataset_format_version__
    assert format_version_match, (
        f"The micro test dataset '{micro_dataset_name}' (located in '{path_dataset_relative}') is outdated. "
        f"The format version for the dataset is '{dataset.info.format_version}', while the current dataset "
        f"format version for the hafnia package is  '{hafnia.__dataset_format_version__}'.\n"
        f"Please rerun this test but set 'force_update=True' to update the micro test dataset."
    )


def test_class_counts_for_task():
    dataset = get_micro_hafnia_dataset(dataset_name="micro-tiny-dataset", force_update=False)
    counts = dataset.class_counts_for_task(primitive=Bbox)
    assert isinstance(counts, dict)
    assert len(counts) == len(dataset.info.tasks[0].class_names)


def test_class_counts_all():
    dataset = get_micro_hafnia_dataset(dataset_name="micro-tiny-dataset", force_update=False)
    counts = dataset.class_counts_all()
    assert isinstance(counts, dict)
    expected_num_classes = sum(len(task.class_names) for task in dataset.info.tasks if task.class_names)
    assert len(counts) == expected_num_classes


def test_print_stats():
    dataset = get_micro_hafnia_dataset(dataset_name="micro-tiny-dataset", force_update=False)
    dataset.print_stats()


def test_print_sample_and_task_counts():
    dataset = get_micro_hafnia_dataset(dataset_name="micro-tiny-dataset", force_update=False)
    dataset.print_sample_and_task_counts()


def test_print_class_distribution():
    dataset = get_micro_hafnia_dataset(dataset_name="micro-tiny-dataset", force_update=False)
    dataset.print_class_distribution()
