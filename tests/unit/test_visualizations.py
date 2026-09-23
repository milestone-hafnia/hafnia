from typing import Callable

import numpy as np
import pytest

from hafnia.dataset import image_visualizations
from hafnia.dataset.hafnia_dataset_types import ClassInfo, TaskInfo
from hafnia.dataset.primitives import Bbox, Bitmask, Classification, KeyPoint, Point, Polygon, Skeleton
from tests import helper_testing


@pytest.mark.parametrize("dataset_name", helper_testing.MICRO_DATASETS)
def test_mask_region(compare_to_expected_image: Callable, dataset_name: str):
    sample = helper_testing.get_sample_micro_hafnia_dataset(dataset_name=dataset_name, force_update=False)
    image = sample.read_image()
    if dataset_name == "micro-coco-2017":
        annotations = sample.get_primitives([Bitmask])
    else:
        annotations = sample.get_primitives()
    masked_image = image_visualizations.draw_masks(image, annotations)
    compare_to_expected_image(masked_image)


@pytest.mark.parametrize("dataset_name", helper_testing.MICRO_DATASETS)
def test_draw_annotations(compare_to_expected_image: Callable, dataset_name: str):
    sample = helper_testing.get_sample_micro_hafnia_dataset(dataset_name=dataset_name, force_update=False)
    image = sample.read_image()
    annotations = sample.get_primitives()
    masked_image = image_visualizations.draw_annotations(image, annotations)
    compare_to_expected_image(masked_image)


@pytest.mark.parametrize("dataset_name", helper_testing.MICRO_DATASETS)
def test_blur_anonymization(compare_to_expected_image: Callable, dataset_name: str):
    sample = helper_testing.get_sample_micro_hafnia_dataset(dataset_name=dataset_name, force_update=False)
    image = sample.read_image()
    if dataset_name == "micro-coco-2017":
        annotations = sample.get_primitives([Bitmask])
    else:
        annotations = sample.get_primitives([Bitmask, Bbox, Polygon])

    masked_image = image_visualizations.draw_anonymize_by_blurring(image, annotations)
    compare_to_expected_image(masked_image)


def test_polygon_to_bitmask_conversion(compare_to_expected_image: Callable):
    sample = helper_testing.get_sample_micro_hafnia_dataset(dataset_name="micro-tiny-dataset", force_update=False)
    image = sample.read_image()
    annotations = sample.get_primitives()
    polygons = [a for a in annotations if isinstance(a, Polygon)]

    bitmasks = []
    assert len(polygons) > 0, "There should be at least one Polygon annotation in the sample to test mask conversion."
    for polygon in polygons:
        bitmask = polygon.to_bitmask(img_height=image.shape[0], img_width=image.shape[1])
        bitmasks.append(bitmask)
        mask_from_polygon = polygon.to_mask(img_height=image.shape[0], img_width=image.shape[1], use_coco_utils=True)
        mask_from_bitmask = bitmask.to_mask(img_height=image.shape[0], img_width=image.shape[1])
        assert np.array_equal(mask_from_polygon, mask_from_bitmask), "Masks from Polygon and Bitmask should match."

    masked_image = image_visualizations.draw_annotations(image, bitmasks)
    compare_to_expected_image(masked_image)


def get_nested_annotation_primitives() -> list:
    """A primitive of each type with nested primitives ('attributes') of its own."""
    color_attribute = Classification(
        class_name="Red",
        task_name="Vehicle Color",
        classifications=[Classification(class_name="Dark", task_name="Color Tone")],  # Nested in a nested primitive
    )
    bbox = Bbox(
        top_left_x=0.05,
        top_left_y=0.1,
        width=0.4,
        height=0.35,
        class_name="Vehicle",
        classifications=[color_attribute, Classification(class_name="No", task_name="Occluded")],
        bboxes=[Bbox(top_left_x=0.1, top_left_y=0.35, width=0.12, height=0.06, class_name="License Plate")],
        keypoints=[KeyPoint(point=Point(x=0.35, y=0.15), class_name="Mirror")],
    )
    polygon = Polygon.from_list_of_points([[0.55, 0.1], [0.9, 0.15], [0.85, 0.4]], class_name="Road")
    polygon.classifications = [Classification(class_name="Wet", task_name="Surface")]

    keypoint = KeyPoint(
        point=Point(x=0.15, y=0.75),
        class_name="Nose",
        classifications=[Classification(class_name="Visible", task_name="Visibility")],
    )
    skeleton = Skeleton(
        keypoints=[KeyPoint(point=Point(x=0.6, y=0.6)), KeyPoint(point=Point(x=0.7, y=0.8))],
        class_name="PersonPose",
        classifications=[Classification(class_name="Standing", task_name="Activity")],
    )
    return [bbox, polygon, keypoint, skeleton]


def test_draw_nested_primitives(compare_to_expected_image: Callable):
    image = np.full((400, 600, 3), 40, dtype=np.uint8)
    image_drawn = image_visualizations.draw_annotations(image, get_nested_annotation_primitives())
    compare_to_expected_image(image_drawn)


def test_draw_nested_primitives_does_not_resize_image():
    """Nested classifications are drawn inside the image and must not append a text banner below it."""
    image = np.full((400, 600, 3), 40, dtype=np.uint8)
    for primitive in get_nested_annotation_primitives():
        assert primitive.draw(image).shape == image.shape


def test_draw_nested_primitives_changes_the_image():
    image = np.full((200, 200, 3), 40, dtype=np.uint8)
    bbox_without_attributes = Bbox(top_left_x=0.1, top_left_y=0.1, width=0.5, height=0.5, class_name="Vehicle")
    bbox_with_attributes = bbox_without_attributes.model_copy(
        update={"classifications": [Classification(class_name="Red", task_name="Vehicle Color")]}
    )
    assert not np.array_equal(bbox_without_attributes.draw(image), bbox_with_attributes.draw(image))


def test_draw_nested_style_differs_from_top_level_style():
    image = np.full((200, 200, 3), 40, dtype=np.uint8)
    bbox = Bbox(top_left_x=0.1, top_left_y=0.1, width=0.5, height=0.5, class_name="Vehicle")
    assert not np.array_equal(bbox.draw(image, nested=False), bbox.draw(image, nested=True))


def test_get_nested_primitives_ignores_the_keypoints_of_a_skeleton():
    """'Skeleton.keypoints' are the vertices of the skeleton itself - not nested primitives."""
    attribute = Classification(class_name="Standing", task_name="Activity")
    skeleton = Skeleton(
        keypoints=[KeyPoint(point=Point(x=0.5, y=0.5))], class_name="PersonPose", classifications=[attribute]
    )
    assert skeleton.get_nested_primitives() == [attribute]


def test_get_nested_task_resolves_the_task_of_an_attribute():
    attribute_task = TaskInfo(primitive=Classification, name="Vehicle Color", classes=[ClassInfo(name="Red")])
    task = TaskInfo(
        primitive=Bbox, name="object_detection", classes=[ClassInfo(name="Vehicle", attributes=[attribute_task])]
    )
    attribute = Classification(class_name="Red", task_name="Vehicle Color")
    bbox = Bbox(
        top_left_x=0.1, top_left_y=0.1, width=0.5, height=0.5, class_name="Vehicle", classifications=[attribute]
    )

    assert bbox.get_nested_task(attribute, task) == attribute_task
    assert bbox.get_nested_task(attribute, None) is None
    unknown_attribute = Classification(class_name="Red", task_name="Unknown Task")
    assert bbox.get_nested_task(unknown_attribute, task) is None


def test_nested_label_line_count():
    """Nested classifications stack their labels, so each occupies a line - including their own attributes."""
    classification = Classification(
        class_name="Red",
        task_name="Vehicle Color",
        classifications=[Classification(class_name="Dark", task_name="Color Tone")],
    )
    assert classification.nested_label_line_count() == 2
    # Primitives with image coordinates draw their label themselves and occupy no line
    assert Bbox(top_left_x=0.1, top_left_y=0.1, width=0.5, height=0.5).nested_label_line_count() == 0


def test_draw_label_false_also_suppresses_nested_labels():
    """'draw_label=False' should leave the image free of text - also for nested primitives.

    'Skeleton.draw' relies on this to draw its keypoints (vertices) without a label each.
    """
    image = np.full((200, 400, 3), 40, dtype=np.uint8)
    attribute = [Classification(class_name="Visible", task_name="Visibility")]
    primitives_without_and_with_attributes = [
        (
            Bbox(top_left_x=0.1, top_left_y=0.1, width=0.5, height=0.5, class_name="Vehicle"),
            Bbox(
                top_left_x=0.1,
                top_left_y=0.1,
                width=0.5,
                height=0.5,
                class_name="Vehicle",
                classifications=attribute,
            ),
        ),
        (
            KeyPoint(point=Point(x=0.5, y=0.5), class_name="Nose"),
            KeyPoint(point=Point(x=0.5, y=0.5), class_name="Nose", classifications=attribute),
        ),
    ]
    for primitive, primitive_with_attributes in primitives_without_and_with_attributes:
        name = type(primitive).__name__
        drawn = primitive.draw(image, draw_label=False)
        drawn_with_attributes = primitive_with_attributes.draw(image, draw_label=False)
        assert np.array_equal(drawn, drawn_with_attributes), f"Nested label was drawn for '{name}'"

        # Sanity check: the attribute is drawn when labels are enabled
        assert not np.array_equal(
            primitive.draw(image, draw_label=True), primitive_with_attributes.draw(image, draw_label=True)
        ), f"Expected the nested label to be drawn for '{name}'"


def test_draw_nested_primitives_without_inplace_keeps_the_image_unchanged():
    """'inplace=False' should draw on a copy - also for a nested 'Classification' drawn at an anchor."""
    image = np.full((200, 400, 3), 40, dtype=np.uint8)
    image_before = image.copy()

    Classification(class_name="Red", task_name="Vehicle Color").draw(image, inplace=False, nested=True, anchor=(10, 50))
    assert np.array_equal(image, image_before), "Expected 'inplace=False' to leave the provided image unchanged"
