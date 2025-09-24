from typing import Type

import numpy as np
import pytest

from hafnia.dataset.dataset_names import FieldName
from hafnia.dataset.hafnia_dataset import Sample
from hafnia.dataset.primitives import PRIMITIVE_TYPES
from hafnia.dataset.primitives.bbox import Bbox
from hafnia.dataset.primitives.bitmask import Bitmask
from hafnia.dataset.primitives.classification import Classification
from hafnia.dataset.primitives.polygon import Polygon
from hafnia.dataset.primitives.primitive import Primitive


def get_initialized_dummy_primitives_using_default_task_name(TypePrimitive: Type[Primitive]) -> Primitive:
    """
    Returns a list of initialized dummy primitives for testing.
    """
    if TypePrimitive == Classification:
        return Classification(class_name="dummy_classification", label="dummy_label")
    elif TypePrimitive == Bbox:
        return Bbox(top_left_x=0.1, top_left_y=0.2, width=0.3, height=0.4, class_name="dummy_bbox")
    elif TypePrimitive == Polygon:
        return Polygon.from_list_of_points(points=[(0.1, 0.2), (0.3, 0.4), (0.5, 0.6)], class_name="dummy_polygon")
    elif TypePrimitive == Bitmask:
        return Bitmask.from_mask(
            mask=np.array([[True, False], [False, True]], dtype=bool),
            top=1,
            left=1,
            class_name="dummy_bitmask",
        )
    else:
        raise ValueError(f"Unsupported primitive type: {TypePrimitive}")


def assert_bbox_is_close(actual: Bbox, expected: Bbox, atol: float = 0.001):
    atol = 0.001
    assert type(expected) is type(actual)
    assert np.isclose(actual.height, expected.height, atol=atol)
    assert np.isclose(actual.width, expected.width, atol=atol)
    assert np.isclose(actual.top_left_x, expected.top_left_x, atol=atol)
    assert np.isclose(actual.top_left_y, expected.top_left_y, atol=atol)


@pytest.mark.parametrize("TypePrimitive", PRIMITIVE_TYPES)
def test_sample_primitive_names(TypePrimitive: Type[Primitive]):
    sample = Sample(file_name="test_image.jpg", width=100, height=100, split="test_split", is_sample=False)

    for expected_field in FieldName.fields():
        assert expected_field in TypePrimitive.__annotations__, (
            f"Expected field '{expected_field}' not found in {{TypePrimitive.__name__}} annotations."
        )

    msgs = (
        f"Naming mismatch for coordinate type '{TypePrimitive.__name__}'. "
        f"The column name defined in 'column_name() -> '{TypePrimitive.column_name()}' ' \n"
        f"does not match an attribute in the '{Sample.__name__}' class. Change either 'Sample' or 'column_name()' to match."
    )
    assert hasattr(sample, TypePrimitive.column_name()), msgs

    primitive = get_initialized_dummy_primitives_using_default_task_name(
        TypePrimitive
    )  # Ensure that the dummy primitive can be initialized without errors
    msg = (
        f"The `task_name` of the initialized primitive doesn't match the default task name "
        f"specified in '{TypePrimitive.__name__}.default_task_name()'. Likely, the `model_post_init` of '{Primitive.__name__}', "
        f"have been overridden in '{TypePrimitive.__name__}'. "
    )
    assert primitive.task_name == TypePrimitive.default_task_name(), msg
