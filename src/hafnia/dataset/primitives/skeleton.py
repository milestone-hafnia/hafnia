from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from pydantic import BaseModel, Field

from hafnia.dataset.primitives.keypoint import KeyPoint
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
    """A set of connected keypoints. Used for e.g. human pose estimation and face landmarks.

    The keypoints are defined per annotation, while the edges ('bones') between keypoints are defined
    once per class by the skeleton template in 'ClassInfo.skeleton' of the dataset tasks. Pass the
    template to 'draw' to also draw the edges of the skeleton.
    """

    # Names should match names in FieldName
    keypoints: List[KeyPoint] = Field(
        description="Keypoints (vertices) of the skeleton, e.g. 'Nose' and 'LeftEye'. Ordered by keypoint index"
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

    @classmethod
    def nested_primitive_fields(cls) -> List[str]:
        # 'Skeleton.keypoints' are the vertices of the skeleton itself - not nested primitives -
        # and are drawn by 'Skeleton.draw' together with the edges between them.
        return [field_name for field_name in super().nested_primitive_fields() if field_name != "keypoints"]

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

    def get_skeleton_template(self, task: Optional["TaskInfo"]) -> Optional[SkeletonTemplate]:
        """Get the skeleton template of the class of this annotation from the provided task."""
        if task is None or self.class_name is None or task.classes is None:
            return None
        class_info = task.get_class_by_name(self.class_name, raise_error=False)
        return class_info.skeleton if class_info else None

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
        """Draw the keypoints of the skeleton and, if a task is provided, the edges between them.

        Args:
            image: Image to draw on.
            inplace: If True, draw directly on the provided image instead of a copy.
            draw_label: If True, draw the class name of the skeleton.
            task: Optional `TaskInfo` of the skeleton. The edges between keypoints are defined per class
                by the skeleton template ('ClassInfo.skeleton') of the task. Without a task, only the
                keypoints are drawn.
            nested: If True, the skeleton is drawn as an attribute of another primitive using a thinner
                and smaller style.
            anchor: Unused. A skeleton draws its label at the position of its own keypoints.
        """
        if not inplace:
            image = image.copy()
        points = self.to_pixel_coordinates(image_shape=image.shape[:2])
        if len(points) == 0:
            return image

        class_name = self.get_class_name()
        color = class_color_by_name(class_name)
        font_scale, thickness = draw_style(nested)
        skeleton_template = self.get_skeleton_template(task)
        edges = skeleton_template.edges if skeleton_template else []
        for edge in edges:
            # The template is defined per class, so it may reference keypoints that are not annotated.
            # Inconsistencies are reported by 'HafniaDataset.check_dataset_skeletons' and skipped when drawing.
            is_valid_edge = (0 <= edge.index_start < len(points)) and (0 <= edge.index_end < len(points))
            if not is_valid_edge:
                continue
            cv2.line(image, pt1=points[edge.index_start], pt2=points[edge.index_end], color=color, thickness=thickness)

        for keypoint in self.keypoints:
            keypoint.draw(image, inplace=True, draw_label=False, nested=nested)

        margin = 5
        top_left = (min(x for x, _ in points), min(y for _, y in points) - margin)
        if draw_label:
            cv2.putText(
                img=image,
                text=class_name,
                org=top_left,
                fontFace=FONT_FACE,
                fontScale=font_scale,
                color=color,
                thickness=thickness,
            )

        # Define anchor to place nested classification labels below the label of the  skeleton
        nested_anchor = (int(top_left[0]), int(top_left[1]) + LABEL_LINE_HEIGHT_NESTED)
        return self.draw_nested_primitives(image, draw_label=draw_label, task=task, anchor=nested_anchor)

    def mask(
        self,
        image: np.ndarray,
        inplace: bool = False,
        color: Optional[Tuple[np.uint8, np.uint8, np.uint8]] = None,
    ) -> np.ndarray:
        # Masking is not implemented for the 'Skeleton' primitive, so the image is returned unchanged.
        return image

    def anonymize_by_blurring(self, image: np.ndarray, inplace: bool = False, max_resolution: int = 20) -> np.ndarray:
        # Anonymization is not implemented for the 'Skeleton' primitive, so the image is returned unchanged
        return image

    def get_class_name(self) -> str:
        return get_class_name(self.class_name, self.class_idx)
