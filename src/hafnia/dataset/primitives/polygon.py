from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from hafnia.dataset.primitives.bitmask import Bitmask
from hafnia.dataset.primitives.point import Point
from hafnia.dataset.primitives.primitive import Primitive
from hafnia.dataset.primitives.utils import class_color_by_name, get_class_name


class Polygon(Primitive):
    # Names should match names in FieldName
    points: List[Point]
    class_name: Optional[str] = None  # This should match the string in 'FieldName.CLASS_NAME'
    class_idx: Optional[int] = None  # This should match the string in 'FieldName.CLASS_IDX'
    object_id: Optional[str] = None  # This should match the string in 'FieldName.OBJECT_ID'
    confidence: Optional[float] = None  # Confidence score (0-1.0) for the primitive, e.g. 0.95 for Bbox
    ground_truth: bool = True  # Whether this is ground truth or a prediction

    task_name: str = ""  # Task name to support multiple Polygon tasks in the same dataset. "" defaults to "polygon"
    meta: Optional[Dict[str, Any]] = None  # This can be used to store additional information about the bitmask

    @staticmethod
    def from_list_of_points(
        points: Sequence[Sequence[float]],
        class_name: Optional[str] = None,
        class_idx: Optional[int] = None,
        object_id: Optional[str] = None,
    ) -> "Polygon":
        list_points = [Point(x=point[0], y=point[1]) for point in points]
        return Polygon(points=list_points, class_name=class_name, class_idx=class_idx, object_id=object_id)

    @staticmethod
    def default_task_name() -> str:
        return "polygon"

    @staticmethod
    def column_name() -> str:
        return "polygons"

    def calculate_area(self) -> float:
        raise NotImplementedError()

    def to_pixel_coordinates(
        self, image_shape: Tuple[int, int], as_int: bool = True, clip_values: bool = True
    ) -> List[Tuple]:
        points = [
            point.to_pixel_coordinates(image_shape=image_shape, as_int=as_int, clip_values=clip_values)
            for point in self.points
        ]
        return points

    def draw(self, image: np.ndarray, inplace: bool = False) -> np.ndarray:
        if not inplace:
            image = image.copy()
        points = np.array(self.to_pixel_coordinates(image_shape=image.shape[:2]))

        bottom_left_idx = np.lexsort((-points[:, 1], points[:, 0]))[0]
        bottom_left_np = points[bottom_left_idx, :]
        margin = 5
        bottom_left = (bottom_left_np[0] + margin, bottom_left_np[1] - margin)

        class_name = self.get_class_name()
        color = class_color_by_name(class_name)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.polylines(image, [points], isClosed=True, color=(0, 255, 0), thickness=2)
        cv2.putText(
            img=image, text=class_name, org=bottom_left, fontFace=font, fontScale=0.75, color=color, thickness=2
        )
        return image

    def anonymize_by_blurring(self, image: np.ndarray, inplace: bool = False, max_resolution: int = 20) -> np.ndarray:
        if not inplace:
            image = image.copy()
        points = np.array(self.to_pixel_coordinates(image_shape=image.shape[:2]))
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        mask = cv2.fillPoly(mask, [points], color=255).astype(bool)
        bitmask = Bitmask.from_mask(mask=mask, top=0, left=0).squeeze_mask()
        image = bitmask.anonymize_by_blurring(image=image, inplace=inplace, max_resolution=max_resolution)

        return image

    def mask(
        self, image: np.ndarray, inplace: bool = False, color: Optional[Tuple[np.uint8, np.uint8, np.uint8]] = None
    ) -> np.ndarray:
        if not inplace:
            image = image.copy()
        points = self.to_pixel_coordinates(image_shape=image.shape[:2])

        if color is None:
            mask = np.zeros_like(image[:, :, 0])
            bitmask = cv2.fillPoly(mask, pts=[np.array(points)], color=255).astype(bool)  # type: ignore[assignment]
            color = tuple(int(value) for value in np.mean(image[bitmask], axis=0))  # type: ignore[assignment]

        cv2.fillPoly(image, [np.array(points)], color=color)
        return image

    def get_class_name(self) -> str:
        return get_class_name(self.class_name, self.class_idx)
