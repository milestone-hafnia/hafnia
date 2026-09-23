import hashlib
from typing import Optional, Tuple, Union

import cv2
import numpy as np

FONT_FACE = cv2.FONT_HERSHEY_SIMPLEX

# Font scale and line thickness of a primitive drawn as a top-level annotation and as a nested annotation.
# Nested annotations are drawn thinner and smaller to keep the top-level annotation the dominant one.
FONT_SCALE = 0.75
THICKNESS = 2
FONT_SCALE_NESTED = 0.5
THICKNESS_NESTED = 1

# Distance between two stacked labels of nested primitives in pixels
LABEL_LINE_HEIGHT_NESTED = 16


def draw_style(nested: bool) -> Tuple[float, int]:
    """Font scale and line thickness used when drawing a primitive as a top-level or a nested annotation."""
    if nested:
        return FONT_SCALE_NESTED, THICKNESS_NESTED
    return FONT_SCALE, THICKNESS


def text_org_from_left_bottom_to_centered(xy_org: tuple, text: str, font, font_scale: float, thickness: int) -> tuple:
    xy_text_size = cv2.getTextSize(text, fontFace=font, fontScale=font_scale, thickness=thickness)[0]
    xy_text_size_half = np.array(xy_text_size) / 2
    xy_centered_np = xy_org + xy_text_size_half * np.array([-1, 1])
    xy_centered = tuple(int(value) for value in xy_centered_np)
    return xy_centered


def round_int_clip_value(value: Union[int, float], max_value: int) -> int:
    return clip(value=int(round(value)), v_min=0, v_max=max_value)  # noqa: RUF046


def class_color_by_name(name: str) -> Tuple[int, int, int]:
    # Create a hash of the class name
    hash_object = hashlib.md5(name.encode())
    # Use the hash to generate a color
    hash_digest = hash_object.hexdigest()
    color = (int(hash_digest[0:2], 16), int(hash_digest[2:4], 16), int(hash_digest[4:6], 16))
    return color


# Define an abstract base class
def clip(value, v_min, v_max):
    return min(max(v_min, value), v_max)


def get_class_name(class_name: Optional[str], class_idx: Optional[int]) -> str:
    if class_name is not None:
        return class_name
    if class_idx is not None:
        return f"IDX:{class_idx}"
    return "NoName"


def anonymize_by_resizing(blur_region: np.ndarray, max_resolution: int = 20) -> np.ndarray:
    """
    Removes high-frequency details from a region of an image by resizing it down and then back up.

    We use downscaling to max_resolution to ensure that anonymization is handled adaptively, so that
    large objects are filtered more aggressively than small objects.
    This is desirable for anonymization as large objects (e.g., a face taking up a significant portion of the image)
    require more aggressive blurring to anonymize, while small objects are already less identifiable and thus
    require less blurring.
    """
    original_shape = blur_region.shape[:2]
    resize_factor = max(original_shape) / max_resolution
    new_size = (int(original_shape[0] / resize_factor), int(original_shape[1] / resize_factor))
    blur_region_downsized = cv2.resize(blur_region, new_size[::-1], interpolation=cv2.INTER_LINEAR)
    blur_region_upsized = cv2.resize(blur_region_downsized, original_shape[::-1], interpolation=cv2.INTER_LINEAR)
    return blur_region_upsized
