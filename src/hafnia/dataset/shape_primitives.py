from __future__ import annotations

import hashlib
from typing import Any, Dict, List, Optional, Sequence, Tuple, Type

import cv2
import numpy as np
import pycocotools.mask as coco_mask
from pydantic import BaseModel

from hafnia.dataset.base_types import Primitive
from hafnia.dataset.colors import get_n_colors


class Classification(Primitive):
    # Names should match names in FieldName
    class_name: Optional[str] = None  # Class name, e.g. "car"
    class_idx: Optional[int] = None  # Class index, e.g. 0 for "car" if it is the first class
    object_id: Optional[str] = None  # Unique identifier for the object, e.g. "12345123"
    confidence: Optional[float] = None  # Confidence score (0-1.0) for the primitive, e.g. 0.95 for Classification
    ground_truth: bool = True  # Whether this is ground truth or a prediction

    draw_label: bool = True  # Whether to draw the label

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

    def draw(self, image: np.ndarray, inplace: bool = False) -> np.ndarray:
        if self.draw_label is False:
            return image
        from hafnia.dataset import image_operations

        class_name = self.get_class_name()
        if self.task_name == self.default_task_name():
            text = class_name
        else:
            text = f"{self.task_name}: {class_name}"
        image = image_operations.append_text_below_frame(image, text=text)

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


class Bbox(Primitive):
    # Names should match names in FieldName
    height: float  # Height of the bounding box as a fraction of the image height, e.g. 0.1 for 10% of the image height
    width: float  # Width of the bounding box as a fraction of the image width, e.g. 0.1 for 10% of the image width
    top_left_x: float  # X coordinate of top-left corner of Bbox as a fraction of the image width, e.g. 0.1 for 10% of the image width
    top_left_y: float  # Y coordinate of top-left corner of Bbox as a fraction of the image height, e.g. 0.1 for 10% of the image height
    class_name: Optional[str] = None  # Class name, e.g. "car"
    class_idx: Optional[int] = None  # Class index, e.g. 0 for "car" if it is the first class
    object_id: Optional[str] = None  # Unique identifier for the object, e.g. "12345123"
    confidence: Optional[float] = None  # Confidence score (0-1.0) for the primitive, e.g. 0.95 for Bbox
    ground_truth: bool = True  # Whether this is ground truth or a prediction
    draw_label: bool = True  # Whether to draw the label on the bounding box

    task_name: str = ""  # Task name to support multiple Bbox tasks in the same dataset. "" defaults to "bboxes"
    meta: Optional[Dict[str, Any]] = None  # This can be used to store additional information about the bitmask

    @staticmethod
    def default_task_name() -> str:
        return "bboxes"

    @staticmethod
    def column_name() -> str:
        return "objects"

    def calculate_area(self) -> float:
        return self.height * self.width

    @staticmethod
    def from_coco(bbox: List, height: int, width: int) -> Bbox:
        """
        Converts a COCO-style bounding box to a Bbox object.
        The bbox is in the format [x_min, y_min, width, height].
        """
        x_min, y_min, bbox_width, bbox_height = bbox
        return Bbox(
            top_left_x=x_min / width,
            top_left_y=y_min / height,
            width=bbox_width / width,
            height=bbox_height / height,
        )

    def to_bbox(self) -> Tuple[float, float, float, float]:
        """
        Converts Bbox to a tuple of (x_min, y_min, width, height) with normalized coordinates.
        Values are floats in the range [0, 1].
        """
        return (self.top_left_x, self.top_left_y, self.width, self.height)

    def to_coco(self, image_height: int, image_width: int) -> Tuple[int, int, int, int]:
        xmin = round_int_clip_value(self.top_left_x * image_width, max_value=image_width)
        bbox_width = round_int_clip_value(self.width * image_width, max_value=image_width)

        ymin = round_int_clip_value(self.top_left_y * image_height, max_value=image_height)
        bbox_height = round_int_clip_value(self.height * image_height, max_value=image_height)

        return xmin, ymin, bbox_width, bbox_height

    def to_pixel_coordinates(
        self, image_shape: Tuple[int, int], as_int: bool = True, clip_values: bool = True
    ) -> Tuple[float, float, float, float]:
        bb_height = self.height * image_shape[0]
        bb_width = self.width * image_shape[1]
        bb_top_left_x = self.top_left_x * image_shape[1]
        bb_top_left_y = self.top_left_y * image_shape[0]
        xmin, ymin, xmax, ymax = bb_top_left_x, bb_top_left_y, bb_top_left_x + bb_width, bb_top_left_y + bb_height

        if as_int:
            xmin, ymin, xmax, ymax = int(round(xmin)), int(round(ymin)), int(round(xmax)), int(round(ymax))  # noqa: RUF046

        if clip_values:
            xmin = clip(value=xmin, v_min=0, v_max=image_shape[1])
            xmax = clip(value=xmax, v_min=0, v_max=image_shape[1])
            ymin = clip(value=ymin, v_min=0, v_max=image_shape[0])
            ymax = clip(value=ymax, v_min=0, v_max=image_shape[0])

        return xmin, ymin, xmax, ymax

    def draw(self, image: np.ndarray, inplace: bool = False) -> np.ndarray:
        if not inplace:
            image = image.copy()
        xmin, ymin, xmax, ymax = self.to_pixel_coordinates(image_shape=image.shape[:2])

        class_name = self.get_class_name()
        color = class_color_by_name(class_name)
        font = cv2.FONT_HERSHEY_SIMPLEX
        margin = 5
        bottom_left = (xmin + margin, ymax - margin)
        if self.draw_label:
            cv2.putText(
                img=image, text=class_name, org=bottom_left, fontFace=font, fontScale=0.75, color=color, thickness=2
            )
        cv2.rectangle(image, pt1=(xmin, ymin), pt2=(xmax, ymax), color=color, thickness=2)

        return image

    def mask(
        self, image: np.ndarray, inplace: bool = False, color: Optional[Tuple[np.uint8, np.uint8, np.uint8]] = None
    ) -> np.ndarray:
        if not inplace:
            image = image.copy()
        xmin, ymin, xmax, ymax = self.to_pixel_coordinates(image_shape=image.shape[:2])

        if color is None:
            color = np.mean(image[ymin:ymax, xmin:xmax], axis=(0, 1)).astype(np.uint8)

        image[ymin:ymax, xmin:xmax] = color
        return image

    def anonymize_by_blurring(self, image: np.ndarray, inplace: bool = False, max_resolution: int = 20) -> np.ndarray:
        if not inplace:
            image = image.copy()
        xmin, ymin, xmax, ymax = self.to_pixel_coordinates(image_shape=image.shape[:2])
        blur_region = image[ymin:ymax, xmin:xmax]
        blur_region_upsized = anonymize_by_resizing(blur_region, max_resolution=max_resolution)
        image[ymin:ymax, xmin:xmax] = blur_region_upsized
        return image

    def get_class_name(self) -> str:
        return get_class_name(self.class_name, self.class_idx)


class Point(BaseModel):
    x: float
    y: float

    def to_pixel_coordinates(
        self, image_shape: Tuple[int, int], as_int: bool = True, clip_values: bool = True
    ) -> Tuple[Any, Any]:
        x = self.x * image_shape[1]
        y = self.y * image_shape[0]

        if as_int:
            x, y = int(round(x)), int(round(y))  # noqa: RUF046

        if clip_values:
            x = clip(value=x, v_min=0, v_max=image_shape[1])
            y = clip(value=y, v_min=0, v_max=image_shape[0])

        return x, y


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
            bitmask = cv2.fillPoly(mask, pts=[np.array(points)], color=255).astype(bool)
            color = tuple(int(value) for value in np.mean(image[bitmask], axis=0))

        cv2.fillPoly(image, [np.array(points)], color=color)
        return image

    def get_class_name(self) -> str:
        return get_class_name(self.class_name, self.class_idx)


class Bitmask(Primitive):
    # Names should match names in FieldName
    top: int  # Bitmask top coordinate in pixels
    left: int  # Bitmask left coordinate in pixels
    height: int  # Bitmask height of the bounding box in pixels
    width: int  # Bitmask width of the bounding box in pixels
    rleString: str  # Run-length encoding (RLE) string for the bitmask region of size (height, width) at (top, left).
    area: Optional[float] = None  # Area of the bitmask in pixels is calculated from the RLE string
    class_name: Optional[str] = None  # This should match the string in 'FieldName.CLASS_NAME'
    class_idx: Optional[int] = None  # This should match the string in 'FieldName.CLASS_IDX'
    object_id: Optional[str] = None  # This should match the string in 'FieldName.OBJECT_ID'
    confidence: Optional[float] = None  # Confidence score (0-1.0) for the primitive, e.g. 0.95 for Bbox
    ground_truth: bool = True  # Whether this is ground truth or a prediction
    draw_label: bool = True  # Whether to draw the label inside the mask

    task_name: str = ""  # Task name to support multiple Bitmask tasks in the same dataset. "" defaults to "bitmask"
    meta: Optional[Dict[str, Any]] = None  # This can be used to store additional information about the bitmask

    @staticmethod
    def default_task_name() -> str:
        return "bitmask"

    @staticmethod
    def column_name() -> str:
        return "bitmasks"

    def calculate_area(self) -> float:
        raise NotImplementedError()

    @staticmethod
    def from_mask(
        mask: np.ndarray,
        top: int,  # Bounding box top coordinate in pixels
        left: int,  # Bounding box left coordinate in pixels
        class_name: Optional[str] = None,  # This should match the string in 'FieldName.CLASS_NAME'
        class_idx: Optional[int] = None,  # This should match the string in 'FieldName.CLASS_IDX'
        object_id: Optional[str] = None,  # This should match the string in 'FieldName.OBJECT_ID') -> "Bitmask":
    ):
        if len(mask.shape) != 2:
            raise ValueError("Bitmask should be a 2-dimensional array.")

        if mask.dtype != "|b1":
            raise TypeError("Bitmask should be an array of boolean values. For numpy array call .astype(bool).")

        h, w = mask.shape[:2]
        area_pixels = np.sum(mask != 0)
        area = area_pixels / (h * w)

        mask_fortran = np.asfortranarray(mask, np.prod(h * w))  # Convert to Fortran order for COCO encoding
        rle_coding = coco_mask.encode(mask_fortran.astype(bool))  # Encode the mask using COCO RLE
        rle_string = rle_coding["counts"].decode("utf-8")  # Convert the counts to string

        return Bitmask(
            top=top,
            left=left,
            height=h,
            width=w,
            area=area,
            rleString=rle_string,
            class_name=class_name,
            class_idx=class_idx,
            object_id=object_id,
        )

    def squeeze_mask(self):
        """
        A mask may have large redundant areas of zeros. This function squeezes the mask to remove those areas.
        """
        region_mask = self.to_region_mask()
        shift_left, last_left = np.flatnonzero(region_mask.sum(axis=0))[[0, -1]]
        shift_top, last_top = np.flatnonzero(region_mask.sum(axis=1))[[0, -1]]
        new_top = self.top + shift_top
        new_left = self.left + shift_left
        new_region_mask = region_mask[shift_top : last_top + 1, shift_left : last_left + 1]

        bitmask_squeezed = Bitmask.from_mask(
            mask=new_region_mask,
            top=new_top,
            left=new_left,
            class_name=self.class_name,
            class_idx=self.class_idx,
            object_id=self.object_id,
        )
        return bitmask_squeezed

    def anonymize_by_blurring(self, image: np.ndarray, inplace: bool = False, max_resolution: int = 20) -> np.ndarray:
        mask_tight = self.squeeze_mask()

        mask_region = mask_tight.to_region_mask()
        region_image = image[
            mask_tight.top : mask_tight.top + mask_tight.height, mask_tight.left : mask_tight.left + mask_tight.width
        ]
        region_image_blurred = anonymize_by_resizing(blur_region=region_image, max_resolution=max_resolution)
        image_mixed = np.where(mask_region[:, :, None], region_image_blurred, region_image)
        image[
            mask_tight.top : mask_tight.top + mask_tight.height, mask_tight.left : mask_tight.left + mask_tight.width
        ] = image_mixed
        return image

    def to_region_mask(self) -> np.ndarray:
        """Returns a binary mask from the RLE string. The masks is only the region of the object and not the full image."""
        rle = {"counts": self.rleString.encode(), "size": [self.height, self.width]}
        mask = coco_mask.decode(rle) > 0
        return mask

    def to_mask(self, img_height: int, img_width: int) -> np.ndarray:
        """Creates a full image mask from the RLE string."""

        region_mask = self.to_region_mask()
        bitmask_np = np.zeros((img_height, img_width), dtype=bool)
        bitmask_np[self.top : self.top + self.height, self.left : self.left + self.width] = region_mask
        return bitmask_np

    def draw(self, image: np.ndarray, inplace: bool = False) -> np.ndarray:
        if not inplace:
            image = image.copy()
        if image.ndim == 2:  # for grayscale/monochromatic images
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        img_height, img_width = image.shape[:2]
        bitmask_np = self.to_mask(img_height=img_height, img_width=img_width)

        class_name = self.get_class_name()
        color = class_color_by_name(class_name)

        # Creates transparent masking with the specified color
        image_masked = image.copy()
        image_masked[bitmask_np] = color
        cv2.addWeighted(src1=image, alpha=0.3, src2=image_masked, beta=0.7, gamma=0, dst=image)

        if self.draw_label:
            # Determines the center of mask
            xy = np.stack(np.nonzero(bitmask_np))
            xy_org = tuple(np.median(xy, axis=1).astype(int))[::-1]

            xy_org = np.median(xy, axis=1).astype(int)[::-1]
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.75
            thickness = 2
            xy_centered = text_org_from_left_bottom_to_centered(xy_org, class_name, font, font_scale, thickness)

            cv2.putText(
                img=image,
                text=class_name,
                org=xy_centered,
                fontFace=font,
                fontScale=font_scale,
                color=(255, 255, 255),
                thickness=thickness,
            )
        return image

    def mask(
        self, image: np.ndarray, inplace: bool = False, color: Optional[Tuple[np.uint8, np.uint8, np.uint8]] = None
    ) -> np.ndarray:
        if not inplace:
            image = image.copy()

        bitmask_np = self.to_mask(img_height=image.shape[0], img_width=image.shape[1])

        if color is None:
            color = tuple(int(value) for value in np.mean(image[bitmask_np], axis=0))
        image[bitmask_np] = color
        return image

    def get_class_name(self) -> str:
        return get_class_name(self.class_name, self.class_idx)


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

        color_mapping = np.array(get_n_colors(len(self.class_names)), dtype=np.uint8)
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


def text_org_from_left_bottom_to_centered(xy_org: tuple, text: str, font, font_scale: float, thickness: int) -> tuple:
    xy_text_size = cv2.getTextSize(text, fontFace=font, fontScale=font_scale, thickness=thickness)[0]
    xy_text_size_half = np.array(xy_text_size) / 2
    xy_centered_np = xy_org + xy_text_size_half * np.array([-1, 1])
    xy_centered = tuple(int(value) for value in xy_centered_np)
    return xy_centered


def round_int_clip_value(value: int, max_value: int) -> int:
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
    """
    original_shape = blur_region.shape[:2]
    resize_factor = max(original_shape) / max_resolution
    new_size = (int(original_shape[0] / resize_factor), int(original_shape[1] / resize_factor))
    blur_region_downsized = cv2.resize(blur_region, new_size[::-1], interpolation=cv2.INTER_LINEAR)
    blur_region_upsized = cv2.resize(blur_region_downsized, original_shape[::-1], interpolation=cv2.INTER_LINEAR)
    return blur_region_upsized


PRIMITIVE_TYPES: List[Type[Primitive]] = [Bbox, Classification, Polygon, Bitmask]
PRIMITIVE_NAME_TO_TYPE = {cls.__name__: cls for cls in PRIMITIVE_TYPES}
PRIMITIVE_COLUMN_NAMES: List[str] = [PrimitiveType.column_name() for PrimitiveType in PRIMITIVE_TYPES]
