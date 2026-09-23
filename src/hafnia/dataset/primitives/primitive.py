from __future__ import annotations

from abc import ABCMeta, abstractmethod
from typing import TYPE_CHECKING, List, Optional, Tuple

import numpy as np
from pydantic import BaseModel

from hafnia.dataset.primitives.utils import LABEL_LINE_HEIGHT_NESTED

if TYPE_CHECKING:  # Using 'TYPE_CHECKING' to avoid circular imports during type checking
    from hafnia.dataset.hafnia_dataset_types import TaskInfo


# Fields that may hold nested primitives ('attributes') of a primitive. The order matches the drawing
# order of 'image_visualizations.draw_annotations' so that nested annotations are layered the same way.
NESTED_PRIMITIVE_FIELDS: Tuple[str, ...] = (
    "bitmasks",
    "bboxes",
    "polygons",
    "skeletons",
    "keypoints",
    "classifications",
)


class Primitive(BaseModel, metaclass=ABCMeta):
    def model_post_init(self, context) -> None:
        if self.task_name == "":  # type: ignore[has-type] # Hack because 'task_name' doesn't exist in base-class yet.
            self.task_name = self.default_task_name()

    @staticmethod
    @abstractmethod
    def default_task_name() -> str:
        # E.g. "return bboxes" for Bbox
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def column_name() -> str:
        """
        Name of field used in hugging face datasets for storing annotations
        E.g. "bboxes" for Bbox.
        """
        pass

    @abstractmethod
    def calculate_area(self, image_height: int, image_width: int) -> float:
        # Calculate the area of the primitive
        pass

    @classmethod
    def nested_primitive_fields(cls) -> List[str]:
        """Fields of this primitive that hold nested primitives ('attributes'), in drawing order."""
        return [field_name for field_name in NESTED_PRIMITIVE_FIELDS if field_name in cls.model_fields]

    def get_nested_primitives(self) -> List[Primitive]:
        """Nested primitives ('attributes') of this primitive, in drawing order."""
        nested_primitives: List[Primitive] = []
        for field_name in self.nested_primitive_fields():
            nested_primitives.extend(getattr(self, field_name) or [])
        return nested_primitives

    def get_nested_task(self, primitive: Primitive, task: Optional[TaskInfo]) -> Optional[TaskInfo]:
        """Find the task of a nested primitive among the attribute tasks of this primitive's class.

        The tasks of nested primitives are defined per class of the parent task ('ClassInfo.attributes').
        Returns 'None' if the task of the nested primitive can not be resolved.
        """
        class_name = getattr(self, "class_name", None)
        if task is None or class_name is None:
            return None

        class_info = task.get_class_by_name(class_name, raise_error=False)
        if class_info is None or class_info.attributes is None:
            return None

        for attribute_task in class_info.attributes:
            if attribute_task.primitive is type(primitive) and attribute_task.name == primitive.task_name:
                return attribute_task
        return None

    def nested_label_line_count(self) -> int:
        """Number of stacked text lines this primitive occupies when it is drawn at an anchor.

        Zero for primitives that are drawn at their own image coordinates instead of at the anchor.
        """
        return 0

    def draw_nested_primitives(
        self,
        image: np.ndarray,
        *,
        draw_label: bool = True,
        task: Optional[TaskInfo] = None,
        anchor: Optional[Tuple[int, int]] = None,
    ) -> np.ndarray:
        """Draw the nested primitives ('attributes') of this primitive with their own 'draw' functions.

        Args:
            image: Image to draw on. Always drawn on inplace.
            draw_label: If True, draw the class names of the nested primitives. Pass on the 'draw_label'
                of this primitive, so that e.g. the keypoints of a `Skeleton` stay free of text.
            task: Optional `TaskInfo` of *this* primitive. The task of each nested primitive is resolved
                from the attribute tasks of the class of this primitive, see 'get_nested_task'.
            anchor: Position where nested primitives without image coordinates (e.g. `Classification`)
                draw their label. The anchor moves down for each label that has been drawn.
        """
        for primitive in self.get_nested_primitives():
            image = primitive.draw(
                image,
                inplace=True,
                draw_label=draw_label,
                nested=True,
                task=self.get_nested_task(primitive, task),
                anchor=anchor,
            )
            if anchor is not None:
                anchor = (anchor[0], anchor[1] + primitive.nested_label_line_count() * LABEL_LINE_HEIGHT_NESTED)
        return image

    @abstractmethod
    def draw(
        self,
        image: np.ndarray,
        inplace: bool = False,
        draw_label: bool = True,
        *,
        task: Optional[TaskInfo] = None,
        nested: bool = False,
        anchor: Optional[Tuple[int, int]] = None,
    ) -> np.ndarray:
        """Draw the primitive and its nested primitives ('attributes') on an image.

        Args:
            image: Image to draw on.
            inplace: If True, draw directly on the provided image instead of a copy.
            draw_label: If True, draw the class name of the primitive.
            task: Optional `TaskInfo` of the primitive, providing class-level information that is not
                stored on the annotation itself. Used by `Skeleton` to draw the edges between keypoints
                as defined by the skeleton template of the class, and to resolve the tasks of nested
                primitives.
            nested: If True, the primitive is drawn as an attribute of another primitive using a thinner
                and smaller style to keep the top-level annotation the dominant one.
            anchor: Position to draw the label at. Only used by primitives without image coordinates
                (`Classification`); primitives with image coordinates place their label themselves.
        """

    @abstractmethod
    def mask(
        self,
        image: np.ndarray,
        inplace: bool = False,
        color: Optional[Tuple[np.uint8, np.uint8, np.uint8]] = None,
    ) -> np.ndarray:
        pass

    @abstractmethod
    def anonymize_by_blurring(self, image: np.ndarray, inplace: bool = False, max_resolution: int = 20) -> np.ndarray:
        pass
