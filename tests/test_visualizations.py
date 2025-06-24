from typing import Callable

import numpy as np
import pytest

from hafnia import helper_testing
from hafnia.dataset.primitives.bbox import Bbox
from hafnia.dataset.primitives.bitmask import Bitmask
from hafnia.dataset.primitives.polygon import Polygon
from hafnia.visualizations import image_visualizations


@pytest.mark.parametrize("dataset_name", helper_testing.MICRO_DATASETS)
def test_mask_region(compare_to_expected_image: Callable, dataset_name: str):
    sample = helper_testing.get_sample_micro_hafnia_dataset(dataset_name=dataset_name, force_update=False)
    image = sample.read_image()
    if dataset_name == "coco-2017":
        annotations = sample.get_annotations([Bitmask])
    else:
        annotations = sample.get_annotations()
    masked_image = image_visualizations.draw_masks(image, annotations)
    compare_to_expected_image(masked_image)


@pytest.mark.parametrize("dataset_name", helper_testing.MICRO_DATASETS)
def test_draw_annotations(compare_to_expected_image: Callable, dataset_name: str):
    sample = helper_testing.get_sample_micro_hafnia_dataset(dataset_name=dataset_name, force_update=False)
    image = sample.read_image()
    annotations = sample.get_annotations()
    masked_image = image_visualizations.draw_annotations(image, annotations)
    compare_to_expected_image(masked_image)


@pytest.mark.parametrize("dataset_name", helper_testing.MICRO_DATASETS)
def test_blur_anonymization(compare_to_expected_image: Callable, dataset_name: str):
    sample = helper_testing.get_sample_micro_hafnia_dataset(dataset_name=dataset_name, force_update=False)
    image = sample.read_image()
    if dataset_name == "coco-2017":
        annotations = sample.get_annotations([Bitmask])
    else:
        annotations = sample.get_annotations([Bitmask, Bbox, Polygon])

    masked_image = image_visualizations.draw_anonymize_by_blurring(image, annotations)
    compare_to_expected_image(masked_image)


def test_bitmask_squeezing():
    sample = helper_testing.get_sample_micro_hafnia_dataset(dataset_name="coco-2017", force_update=False)
    image = sample.read_image()
    annotations = sample.get_annotations()
    bitmasks = [a for a in annotations if isinstance(a, Bitmask)]

    assert len(bitmasks) > 0, "There should be at least one Bitmask annotation in the sample to test squeezing."
    for bitmask in bitmasks:
        bitmask_before_squeeze = bitmask.to_mask(image.shape[0], image.shape[1])
        bitmask_squeezed = bitmask.squeeze_mask()

        bitmask_after_squeeze = bitmask_squeezed.to_mask(image.shape[0], image.shape[1])

        assert np.array_equal(bitmask_before_squeeze, bitmask_after_squeeze), (
            "Bitmask before and after squeezing should be equal"
        )
