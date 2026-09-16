from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from pydantic import BaseModel, Field

from hafnia.dataset.primitives.keypoint import KeyPoint
from hafnia.dataset.primitives.primitive import Primitive
from hafnia.dataset.primitives.utils import class_color_by_name, get_class_name

if TYPE_CHECKING:
    from hafnia.dataset.primitives import Classification


class SkeletonEdge(BaseModel):
    """A connection ('bone') between two keypoints of a 'Skeleton' defined by their keypoint index."""

    index_start: int = Field(description="Index of the start keypoint in 'Skeleton.keypoints'")
    index_end: int = Field(description="Index of the end keypoint in 'Skeleton.keypoints'")


class SkeletonTemplate(BaseModel):
    """The keypoints and edges defining the structure of a 'Skeleton' class, e.g. a human pose or a face.

    Stored in 'ClassInfo.skeleton' as the canonical definition of a skeleton class. Note that the
    template is defined per class, as e.g. a body pose and a face have a different set of keypoints.
    """

    keypoint_names: List[str] = Field(
        description="Keypoint names ordered by keypoint index, e.g. ['Nose', 'LeftEye', 'RightEye']"
    )
    edges: List[SkeletonEdge] = Field(
        default_factory=list, description="Connections ('bones') between the keypoints of the skeleton"
    )


class Skeleton(Primitive):
    """A set of connected keypoints. Used for e.g. human pose estimation and face landmarks."""

    # Names should match names in FieldName
    keypoints: List[KeyPoint] = Field(
        description="Keypoints (vertices) of the skeleton, e.g. 'Nose' and 'LeftEye'. Ordered by keypoint index"
    )
    edges: Optional[List[SkeletonEdge]] = Field(
        default=None, description="Connections ('bones') between keypoints as defined by the skeleton template"
    )
    class_name: Optional[str] = Field(default=None, description="Class name of the skeleton, e.g. 'person_pose'")
    class_idx: Optional[int] = Field(default=None, description="Class index of the skeleton")
    object_id: Optional[str] = Field(default=None, description="Object ID of the skeleton")
    confidence: float = Field(
        default=1.0, description="Confidence score (0-1.0) for the primitive, e.g. 0.95 for Skeleton"
    )
    ground_truth: bool = Field(default=True, description="Whether this is ground truth or a prediction")

    task_name: str = Field(
        default="",
        description="Task name to support multiple Skeleton tasks in the same dataset. Defaults to 'pose_estimation'",
    )
    created_at: Optional[datetime] = Field(default=None, description="Date when the primitive was created")
    updated_at: Optional[datetime] = Field(default=None, description="Date when the primitive was last updated")
    meta: Optional[Dict[str, Any]] = Field(
        default=None, description="This can be used to store additional information about the skeleton"
    )

    # Attributes
    classifications: Optional[List["Classification"]] = None

    @staticmethod
    def default_task_name() -> str:
        return "pose_estimation"

    @staticmethod
    def column_name() -> str:
        return "skeletons"

    def calculate_area(self, image_height: int, image_width: int) -> float:
        """A skeleton consists of keypoints with no spatial extent and therefore no area."""
        return 0.0

    def to_pixel_coordinates(
        self, image_shape: Tuple[int, int], as_int: bool = True, clip_values: bool = True
    ) -> List[Tuple]:
        return [
            keypoint.to_pixel_coordinates(image_shape=image_shape, as_int=as_int, clip_values=clip_values)
            for keypoint in self.keypoints
        ]

    def draw(self, image: np.ndarray, inplace: bool = False, draw_label: bool = True) -> np.ndarray:
        if not inplace:
            image = image.copy()
        points = self.to_pixel_coordinates(image_shape=image.shape[:2])

        class_name = self.get_class_name()
        color = class_color_by_name(class_name)
        for edge in self.edges or []:
            cv2.line(image, pt1=points[edge.index_start], pt2=points[edge.index_end], color=color, thickness=2)

        for keypoint in self.keypoints:
            keypoint.draw(image, inplace=True, draw_label=False)

        if draw_label:
            margin = 5
            top_left = (min(x for x, _ in points), min(y for _, y in points) - margin)
            cv2.putText(
                img=image,
                text=class_name,
                org=top_left,
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
        raise NotImplementedError("Masking is not supported for the 'Skeleton' primitive")

    def anonymize_by_blurring(self, image: np.ndarray, inplace: bool = False, max_resolution: int = 20) -> np.ndarray:
        raise NotImplementedError("Anonymization by blurring is not supported for the 'Skeleton' primitive")

    def get_class_name(self) -> str:
        return get_class_name(self.class_name, self.class_idx)
