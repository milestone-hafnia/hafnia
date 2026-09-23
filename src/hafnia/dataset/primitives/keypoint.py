from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from pydantic import Field

from hafnia.dataset.primitives.point import Point
from hafnia.dataset.primitives.primitive import Primitive
from hafnia.dataset.primitives.utils import (
    FONT_FACE,
    LABEL_LINE_HEIGHT_NESTED,
    class_color_by_name,
    draw_style,
    get_class_name,
)

if TYPE_CHECKING:
    from hafnia.dataset.hafnia_dataset_types import TaskInfo
    from hafnia.dataset.primitives import Classification


class KeyPoint(Primitive):
    # Names should match names in FieldName
    point: Point = Field(description="Normalized point (x, y) defining the keypoint location")
    labeled: bool = Field(
        default=True,
        description=(
            "Whether the keypoint has been labeled. An unlabeled keypoint has no meaningful location, but is "
            "kept to match the keypoints of the skeleton template of its class. Unlabeled keypoints - and the "
            "skeleton edges towards them - are skipped when drawing"
        ),
    )
    class_name: Optional[str] = Field(default=None, description="Class name of the keypoint, e.g. 'left_eye'")
    class_idx: Optional[int] = Field(default=None, description="Class index of the keypoint")
    object_id: Optional[str] = Field(default=None, description="Object ID of the keypoint")
    confidence: float = Field(
        default=1.0, description="Confidence score (0-1.0) for the primitive, e.g. 0.95 for KeyPoint"
    )
    ground_truth: bool = Field(default=True, description="Whether this is ground truth or a prediction")

    task_name: str = Field(
        default="",
        description="Task name to support multiple KeyPoint tasks in the same dataset. Defaults to 'keypoint_detection'",
    )
    created_at: Optional[datetime] = Field(default=None, description="Date when the primitive was created")
    updated_at: Optional[datetime] = Field(default=None, description="Date when the primitive was last updated")
    meta: Optional[Dict[str, Any]] = Field(
        default=None, description="This can be used to store additional information about the keypoint"
    )

    # Attributes
    classifications: Optional[List["Classification"]] = None

    @staticmethod
    def default_task_name() -> str:
        return "keypoint_detection"

    @staticmethod
    def column_name() -> str:
        return "keypoints"

    def calculate_area(self, image_height: int, image_width: int) -> float:
        """A keypoint has no spatial extent and therefore no area."""
        return 0.0

    def to_pixel_coordinates(
        self, image_shape: Tuple[int, int], as_int: bool = True, clip_values: bool = True
    ) -> Tuple[Any, Any]:
        return self.point.to_pixel_coordinates(image_shape=image_shape, as_int=as_int, clip_values=clip_values)

    def draw(
        self,
        image: np.ndarray,
        inplace: bool = False,
        draw_label: bool = True,
        *,
        task: Optional["TaskInfo"] = None,
        nested: bool = False,
        anchor: Optional[Tuple[int, int]] = None,
    ) -> np.ndarray:
        if not inplace:
            image = image.copy()
        if not self.labeled:
            # An unlabeled keypoint has no meaningful location, so there is nothing to draw
            return image
        x, y = self.to_pixel_coordinates(image_shape=image.shape[:2])

        class_name = self.get_class_name()
        color = class_color_by_name(class_name)
        font_scale, thickness = draw_style(nested)
        radius = 3 if nested else 4
        margin = 5
        cv2.circle(image, center=(x, y), radius=radius, color=color, thickness=-1)
        label_org = (x + radius + margin, y - margin)
        if draw_label:
            cv2.putText(
                img=image,
                text=class_name,
                org=label_org,
                fontFace=FONT_FACE,
                fontScale=font_scale,
                color=color,
                thickness=thickness,
            )

        # Define anchor to place nested classification labels below the label of the  keypoint
        nested_anchor = (label_org[0], label_org[1] + LABEL_LINE_HEIGHT_NESTED)
        return self.draw_nested_primitives(image, draw_label=draw_label, task=task, anchor=nested_anchor)

    def mask(
        self,
        image: np.ndarray,
        inplace: bool = False,
        color: Optional[Tuple[np.uint8, np.uint8, np.uint8]] = None,
    ) -> np.ndarray:
        # Masking is not implemented for the 'KeyPoint' primitive, so the image is returned unchanged.
        # A no-op (as for 'Classification') keeps masking of other primitives in a sample working.
        return image

    def anonymize_by_blurring(self, image: np.ndarray, inplace: bool = False, max_resolution: int = 20) -> np.ndarray:
        # Anonymization is not implemented for the 'KeyPoint' primitive, so the image is returned unchanged
        return image

    def get_class_name(self) -> str:
        return get_class_name(self.class_name, self.class_idx)
