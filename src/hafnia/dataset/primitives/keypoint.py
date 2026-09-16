from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from pydantic import Field

from hafnia.dataset.primitives.point import Point
from hafnia.dataset.primitives.primitive import Primitive
from hafnia.dataset.primitives.utils import class_color_by_name, get_class_name

if TYPE_CHECKING:
    from hafnia.dataset.primitives import Classification


class KeyPoint(Primitive):
    # Names should match names in FieldName
    point: Point = Field(description="Normalized point (x, y) defining the keypoint location")
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

    def draw(self, image: np.ndarray, inplace: bool = False, draw_label: bool = True) -> np.ndarray:
        if not inplace:
            image = image.copy()
        x, y = self.to_pixel_coordinates(image_shape=image.shape[:2])

        class_name = self.get_class_name()
        color = class_color_by_name(class_name)
        radius = 4
        cv2.circle(image, center=(x, y), radius=radius, color=color, thickness=-1)
        if draw_label:
            margin = 5
            cv2.putText(
                img=image,
                text=class_name,
                org=(x + radius + margin, y - margin),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.75,
                color=color,
                thickness=2,
            )
        return image

    def mask(
        self,
        image: np.ndarray,
        inplace: bool = False,
        color: Optional[Tuple[np.uint8, np.uint8, np.uint8]] = None,
    ) -> np.ndarray:
        raise NotImplementedError("Masking is not supported for the 'KeyPoint' primitive")

    def anonymize_by_blurring(self, image: np.ndarray, inplace: bool = False, max_resolution: int = 20) -> np.ndarray:
        raise NotImplementedError("Anonymization by blurring is not supported for the 'KeyPoint' primitive")

    def get_class_name(self) -> str:
        return get_class_name(self.class_name, self.class_idx)
