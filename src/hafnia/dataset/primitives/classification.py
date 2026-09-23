from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from pydantic import Field

from hafnia.dataset.primitives.primitive import Primitive

if TYPE_CHECKING:  # Using 'TYPE_CHECKING' to avoid circular imports during type checking
    from hafnia.dataset.hafnia_dataset_types import TaskInfo
from hafnia.dataset.primitives.utils import (
    FONT_FACE,
    LABEL_LINE_HEIGHT_NESTED,
    anonymize_by_resizing,
    class_color_by_name,
    draw_style,
    get_class_name,
)


class Classification(Primitive):
    # Names should match names in FieldName
    class_name: Optional[str] = Field(default=None, description="Class name, e.g. 'car'")
    class_idx: Optional[int] = Field(default=None, description="Class index, e.g. 0 for 'car' if it is the first class")
    object_id: Optional[str] = Field(default=None, description="Unique identifier for the object, e.g. '12345123'")
    confidence: float = Field(
        default=1.0, description="Confidence score (0-1.0) for the primitive, e.g. 0.95 for Classification"
    )
    ground_truth: bool = Field(default=True, description="Whether this is ground truth or a prediction")

    task_name: str = Field(
        default="",
        description="To support multiple Classification tasks in the same dataset. '' defaults to 'classification'",
    )
    created_at: Optional[datetime] = Field(default=None, description="Date when the primitive was created")
    updated_at: Optional[datetime] = Field(default=None, description="Date when the primitive was last updated")
    meta: Optional[Dict[str, Any]] = Field(
        default=None, description="This can be used to store additional information about the classification"
    )

    # Attributes - allow nesting
    classifications: Optional[List["Classification"]] = None

    @staticmethod
    def default_task_name() -> str:
        return "image_classification"

    @staticmethod
    def column_name() -> str:
        return "classifications"

    def calculate_area(self, image_height: int, image_width: int) -> float:
        return 1.0

    def get_label_text(self) -> str:
        class_name = self.get_class_name()
        if self.task_name == self.default_task_name():
            return class_name
        return f"{self.task_name}: {class_name}"

    def nested_label_line_count(self) -> int:
        """A classification occupies one line for itself plus the lines of its nested classifications."""
        return 1 + sum(classification.nested_label_line_count() for classification in self.classifications or [])

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
        """Draw the classification and its nested classifications ('attributes').

        A classification has no image coordinates, so it is drawn as text. As a top-level annotation the
        text is appended below the frame, while a nested classification is drawn at 'anchor' - typically
        the label position of the primitive it is an attribute of.
        """
        if draw_label is False:
            return image

        text = self.get_label_text()
        if nested and anchor is not None:
            if not inplace:
                image = image.copy()
            font_scale, thickness = draw_style(nested=True)
            cv2.putText(
                img=image,
                text=text,
                org=anchor,
                fontFace=FONT_FACE,
                fontScale=font_scale,
                color=class_color_by_name(self.get_class_name()),
                thickness=thickness,
            )
            # Own attributes are indented one level and stacked below this label
            nested_anchor = (anchor[0], anchor[1] + LABEL_LINE_HEIGHT_NESTED)
            return self.draw_nested_primitives(image, draw_label=draw_label, task=task, anchor=nested_anchor)

        from hafnia.dataset import image_visualizations

        image = image_visualizations.append_text_below_frame(image, text=text, text_size_ratio=0.05)
        return self.draw_nested_primitives(image, draw_label=draw_label, task=task, anchor=None)

    def mask(
        self, image: np.ndarray, inplace: bool = False, color: Optional[Tuple[np.uint8, np.uint8, np.uint8]] = None
    ) -> np.ndarray:
        # Classification does not have a mask effect, so we return the image as is
        return image

    def anonymize_by_blurring(self, image: np.ndarray, inplace: bool = False, max_resolution: int = 20) -> np.ndarray:
        # Classification does not have a blur effect, so we return the image as is
        return anonymize_by_resizing(image, max_resolution=max_resolution)

    def get_class_name(self) -> str:
        return get_class_name(self.class_name, self.class_idx)
