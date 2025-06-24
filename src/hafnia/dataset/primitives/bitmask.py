from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np
import pycocotools.mask as coco_mask

from hafnia.dataset.primitives.primitive import Primitive
from hafnia.dataset.primitives.utils import (
    anonymize_by_resizing,
    class_color_by_name,
    get_class_name,
    text_org_from_left_bottom_to_centered,
)


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

    def draw(self, image: np.ndarray, inplace: bool = False, draw_label: bool = True) -> np.ndarray:
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

        if draw_label:
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
            color = tuple(int(value) for value in np.mean(image[bitmask_np], axis=0))  # type: ignore[assignment]
        image[bitmask_np] = color
        return image

    def get_class_name(self) -> str:
        return get_class_name(self.class_name, self.class_idx)
