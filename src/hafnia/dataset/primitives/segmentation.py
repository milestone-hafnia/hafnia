from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from hafnia.dataset.primitives.primitive import Primitive
from hafnia.dataset.primitives.utils import get_class_name
from hafnia.visualizations.colors import get_n_colors


class Segmentation(Primitive):
    # mask: np.ndarray
    class_names: Optional[List[str]] = None  # This should match the string in 'FieldName.CLASS_NAME'
    ground_truth: bool = True  # Whether this is ground truth or a prediction

    # confidence: Optional[float] = None  # Confidence score (0-1.0) for the primitive, e.g. 0.95 for Classification
    task_name: str = (
        ""  # Task name to support multiple Segmentation tasks in the same dataset. "" defaults to "segmentation"
    )
    meta: Optional[Dict[str, Any]] = None  # This can be used to store additional information about the bitmask

    @staticmethod
    def default_task_name() -> str:
        return "segmentation"

    @staticmethod
    def column_name() -> str:
        return "segmentation"

    def calculate_area(self) -> float:
        raise NotImplementedError()

    def draw(self, image: np.ndarray, inplace: bool = False) -> np.ndarray:
        if not inplace:
            image = image.copy()

        color_mapping = np.asarray(get_n_colors(len(self.class_names)), dtype=np.uint8)  # type: ignore[arg-type]
        label_image = color_mapping[self.mask]
        blended = cv2.addWeighted(image, 0.5, label_image, 0.5, 0)
        return blended

    def mask(
        self, image: np.ndarray, inplace: bool = False, color: Optional[Tuple[np.uint8, np.uint8, np.uint8]] = None
    ) -> np.ndarray:
        return image

    def anonymize_by_blurring(self, image: np.ndarray, inplace: bool = False, max_resolution: int = 20) -> np.ndarray:
        return image

    def get_class_name(self) -> str:
        return get_class_name(self.class_name, self.class_idx)
