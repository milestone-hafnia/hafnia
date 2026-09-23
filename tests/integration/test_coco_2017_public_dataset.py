import pytest

from hafnia.dataset.dataset_names import SplitName
from hafnia.dataset.format_conversions.format_coco_2017 import (
    DATASET_NAME,
    coco_2017_as_hafnia_dataset,
)
from hafnia.dataset.format_conversions.public_datasets import public_dataset_to_hafnia_converters
from hafnia.dataset.primitives import Bbox, Bitmask, Skeleton
from tests.helper_testing import is_github_actions_pipeline


def test_coco_2017_is_a_public_dataset() -> None:
    assert DATASET_NAME in public_dataset_to_hafnia_converters()


@pytest.mark.slow
def test_coco_2017_public_dataset() -> None:
    """Convert the public COCO 2017 dataset.

    The first run downloads ~1 GB of images and annotations from 'https://cocodataset.org'. The
    downloaded files are cached on disk, so following runs only pay the conversion cost.
    """
    if is_github_actions_pipeline():
        pytest.skip("Skipping public dataset tests in GitHub Actions to avoid large downloads.")

    dataset = coco_2017_as_hafnia_dataset(n_samples=20, splits=[SplitName.VAL])
    n_samples = 20

    assert len(dataset) == n_samples
    assert dataset.info.dataset_name == DATASET_NAME

    # Object detection, instance segmentation and human pose annotations are all converted
    for primitive in [Bbox, Bitmask, Skeleton]:
        task = dataset.info.get_task_by_primitive(primitive)
        assert task.classes is not None and len(task.classes) > 0

    # 'check_splits=False' as only the validation split of COCO 2017 is converted for now
    dataset.check_dataset(check_splits=False)
