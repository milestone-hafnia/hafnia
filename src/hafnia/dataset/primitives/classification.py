from typing import Any, Dict, Optional, Tuple

import numpy as np

from hafnia.dataset.primitives.primitive import Primitive
from hafnia.dataset.primitives.utils import anonymize_by_resizing, get_class_name


class Classification(Primitive):
    # Names should match names in FieldName
    class_name: Optional[str] = None  # Class name, e.g. "car"
    class_idx: Optional[int] = None  # Class index, e.g. 0 for "car" if it is the first class
    object_id: Optional[str] = None  # Unique identifier for the object, e.g. "12345123"
    confidence: Optional[float] = None  # Confidence score (0-1.0) for the primitive, e.g. 0.95 for Classification
    ground_truth: bool = True  # Whether this is ground truth or a prediction

    task_name: str = ""  # To support multiple Classification tasks in the same dataset. "" defaults to "classification"
    meta: Optional[Dict[str, Any]] = None  # This can be used to store additional information about the bitmask

    @staticmethod
    def default_task_name() -> str:
        return "classification"

    @staticmethod
    def column_name() -> str:
        return "classifications"

    def calculate_area(self) -> float:
        return 1.0

    def draw(self, image: np.ndarray, inplace: bool = False, draw_label: bool = True) -> np.ndarray:
        if draw_label is False:
            return image
        from hafnia.visualizations import image_visualizations

        class_name = self.get_class_name()
        if self.task_name == self.default_task_name():
            text = class_name
        else:
            text = f"{self.task_name}: {class_name}"
        image = image_visualizations.append_text_below_frame(image, text=text)

        return image

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
