from hafnia.dataset.primitives.bbox import Bbox
from tests.helper_testing import get_micro_hafnia_dataset


def test_class_counts_for_task():
    dataset = get_micro_hafnia_dataset(dataset_name="tiny-dataset", force_update=False)
    counts = dataset.class_counts_for_task(primitive=Bbox)
    assert isinstance(counts, dict)
    assert len(counts) == len(dataset.info.tasks[0].class_names)


def test_class_counts_all():
    dataset = get_micro_hafnia_dataset(dataset_name="tiny-dataset", force_update=False)
    counts = dataset.class_counts_all()
    assert isinstance(counts, dict)
    expected_num_classes = sum(len(task.class_names) for task in dataset.info.tasks if task.class_names)
    assert len(counts) == expected_num_classes


def test_print_stats():
    dataset = get_micro_hafnia_dataset(dataset_name="tiny-dataset", force_update=False)
    dataset.print_stats()


def test_print_sample_and_task_counts():
    dataset = get_micro_hafnia_dataset(dataset_name="tiny-dataset", force_update=False)
    dataset.print_sample_and_task_counts()


def test_print_class_distribution():
    dataset = get_micro_hafnia_dataset(dataset_name="tiny-dataset", force_update=False)
    dataset.print_class_distribution()
