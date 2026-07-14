import math
from typing import TYPE_CHECKING, List, NamedTuple, Optional, Tuple

import more_itertools
import polars as pl

from hafnia.dataset.dataset_names import PrimitiveField, SampleField
from hafnia.dataset.primitives import Bbox, Polygon
from hafnia.log import user_logger
from hafnia.utils import progress_bar

if TYPE_CHECKING:
    from hafnia.dataset.hafnia_dataset import HafniaDataset

# A bbox corner this close (in pixels) to a polygon edge is treated as lying on the boundary, i.e. inside.
# Without this tolerance a corner that sits a fraction of a pixel outside the mask counts as "outside",
# which prevents a box that is otherwise fully inside the mask from being dropped and instead leaves it
# either unadjusted or collapsed to a degenerate sliver.
_BOUNDARY_TOLERANCE_PX = 1.0

# Two points closer than this (in pixels squared, via the cross product) are treated as collinear in the
# exact on-segment test. Axis-aligned edges give an exact zero, so this only guards against float noise.
_COLLINEAR_EPS_PX = 1e-6

_OPPOSITE_SIDE = {"top": "bottom", "right": "left", "bottom": "top", "left": "right"}

# Key under a bbox's `meta` dict where adjustment provenance is recorded (only on boxes that were adjusted).
_ADJUSTMENT_META_KEY = "mask_adjustment"

PixelPoint = Tuple[float, float]  # (x, y) in pixels
PixelPolygon = List[PixelPoint]  # polygon vertices in pixels


class PixelBbox(NamedTuple):
    x1: float  # left
    y1: float  # top
    x2: float  # right
    y2: float  # bottom


def adjust_bboxes_from_polygon_masks_dataset(
    dataset: "HafniaDataset",
    polygon_class_names: List[str],
    run_checks: bool = True,
) -> "HafniaDataset":
    """
    Adjust bounding boxes to avoid overlapping with polygon masks for all samples in the dataset.

    The geometry runs in pixel coordinates on lightweight native types (see ``_PixelBox``/``_PixelPolygon``):
    each box and polygon is converted to pixels once, adjusted, and converted back to a normalized ``Bbox``
    on the way out. Working in pixels keeps the boundary tolerance and per-pixel stepping simple, and
    avoiding per-step primitive copies makes the hot loop noticeably faster than carrying primitives
    throughout. Boxes that overlap a mask are shrunk (or dropped); each shrunk box records the adjustment
    under ``meta[_ADJUSTMENT_META_KEY]``.
    """
    if run_checks:
        # Check tasks for 'polygon_class_names'
        polygon_tasks = dataset.info.get_tasks_by_primitive(Polygon)
        if len(polygon_tasks) == 0:
            raise ValueError("No Polygon tasks found in the dataset, cannot adjust bboxes from polygon masks")
        classes_from_tasks = set(more_itertools.flatten([t.get_class_names() or [] for t in polygon_tasks]))
        has_existing_polygon_class = set(polygon_class_names).issubset(classes_from_tasks)
        if not has_existing_polygon_class:
            raise ValueError(
                f"Polygon class names {polygon_class_names} are not present in the dataset tasks. "
                f"Available polygon class names from tasks: {classes_from_tasks}"
            )

        # Check samples for 'polygon_class_names'
        classes_in_samples_df = dataset.create_primitive_table(Polygon)
        if classes_in_samples_df is None:
            raise ValueError(
                "No polygon primitives found in the dataset samples, cannot adjust bboxes from polygon masks"
            )
        classes_in_samples = set(classes_in_samples_df[PrimitiveField.CLASS_NAME].drop_nulls().to_list())
        has_existing_polygon_class_in_samples = len(classes_in_samples.intersection(polygon_class_names)) > 0
        if not has_existing_polygon_class_in_samples:
            raise ValueError(
                f"Polygon class names {polygon_class_names} are not present in the dataset samples. "
                f"Available polygon class names in samples: {classes_in_samples}"
            )

    adjusted_bboxes_per_sample = []
    for sample in progress_bar(dataset, description="Adjusting bboxes"):
        bboxes_dict = sample.get(SampleField.BBOXES, []) or []  # Returns list if missing or is None
        boxes = [Bbox(**bbox) for bbox in bboxes_dict]
        polygons_dict = sample.get(SampleField.POLYGONS, []) or []  # Returns list if missing or is None
        polygons = [Polygon(**poly) for poly in polygons_dict if poly[PrimitiveField.CLASS_NAME] in polygon_class_names]

        adjusted_boxes = _adjust_bboxes_from_polygon_masks(
            boxes=boxes,
            polygons=polygons,
            image_width=sample[SampleField.WIDTH],
            image_height=sample[SampleField.HEIGHT],
        )
        adjusted_boxes_dicts = {SampleField.BBOXES: [box.model_dump(mode="json") for box in adjusted_boxes]}
        adjusted_bboxes_per_sample.append(adjusted_boxes_dicts)  # Convert to list of dicts for JSON serialization
    samples_adjusted_bboxes = dataset.samples.with_columns(pl.from_records(adjusted_bboxes_per_sample))

    if run_checks:
        adjusted_samples = samples_adjusted_bboxes[SampleField.BBOXES] != dataset.samples[SampleField.BBOXES]
        num_adjusted = adjusted_samples.sum()
        user_logger.info(f"Adjusted bboxes for '{num_adjusted}' out of '{len(adjusted_samples)}' samples ")
    dataset_updated = dataset.update_samples(samples_adjusted_bboxes)
    return dataset_updated


def _adjust_bboxes_from_polygon_masks(
    boxes: List[Bbox],
    polygons: List[Polygon],
    image_width: int,
    image_height: int,
) -> List[Bbox]:
    """Adjust bounding boxes to avoid overlapping with polygon masks.

    The geometry runs in pixel coordinates on lightweight native types (see ``_PixelBox``): each box and
    polygon is converted to pixels once, adjusted, and converted back to a ``Bbox``. Boxes that are not
    adjusted at all are returned as an unchanged copy (no pixel round-trip); adjusted boxes get new
    geometry plus an ``_ADJUSTMENT_META_KEY`` entry under ``meta`` recording the adjustment (see
    ``_pixels_to_bbox``).
    """
    polygons_px = [_polygon_to_pixels(polygon, image_width, image_height) for polygon in polygons]

    bboxes_adjusted: List[Bbox] = []
    for bbox in boxes:
        original_box = _bbox_to_pixels(bbox, image_width, image_height)
        box = original_box
        dropped = False
        for polygon_px in polygons_px:
            adjusted = _adjust_box_with_polygon(box, polygon_px)
            if adjusted is None:  # box lies (almost) fully inside a mask -> remove it
                dropped = True
                break
            box = adjusted
        if dropped:
            continue
        if box == original_box:
            # Not adjusted: keep the exact values (no pixel round-trip), but return a distinct object so
            # callers that mutate the result (e.g. relabeling) don't touch the input box.
            bboxes_adjusted.append(bbox.model_copy())
        else:
            bboxes_adjusted.append(_pixels_to_bbox(box, original_box, bbox, image_width, image_height))
    return bboxes_adjusted


def _adjust_box_with_polygon(box: PixelBbox, polygon_px: PixelPolygon) -> Optional[PixelBbox]:
    if len(polygon_px) == 0:
        return box

    # Drop box if all corners are inside the polygon. A boundary tolerance is used here (and only here)
    # so that a box lying almost entirely within the mask - with a corner only a fraction of a pixel
    # outside the edge - is treated as fully inside and dropped, rather than left overlapping the mask or
    # collapsed to a degenerate sliver by the side adjustments below.
    if all(_corners_states(box, polygon_px, tolerance_px=_BOUNDARY_TOLERANCE_PX)):
        return None

    for _ in range(8):
        side = _first_adjacent_pair_inside(box, polygon_px)
        if side is None:
            break
        adjusted = _adjust_side_minimal(box, side, polygon_px)
        if adjusted != box:
            box = adjusted
            continue
        adjusted = _adjust_side_minimal(box, _OPPOSITE_SIDE[side], polygon_px)
        if adjusted != box:
            box = adjusted
            continue
        break

    return box


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _bbox_to_pixels(bbox: Bbox, W: int, H: int) -> PixelBbox:
    x1, y1 = bbox.top_left_x * W, bbox.top_left_y * H
    return PixelBbox(x1=x1, y1=y1, x2=x1 + bbox.width * W, y2=y1 + bbox.height * H)


def _polygon_to_pixels(polygon: Polygon, W: int, H: int) -> PixelPolygon:
    return [(point.x * W, point.y * H) for point in polygon.points]


def _pixels_to_bbox(box: PixelBbox, original_box: PixelBbox, template: Bbox, W: int, H: int) -> Bbox:
    """Clamp ``box`` to the image and convert back to a ``Bbox``, recording the adjustment under ``meta``.

    The recorded metadata captures that the box was adjusted, its original normalized geometry, and
    ``area_ratio`` - the fraction of the original area that remains (1.0 = unchanged, -> 0.0 = shrunk away).
    """
    x1 = _clamp(box.x1, 0.0, W)
    y1 = _clamp(box.y1, 0.0, H)
    x2 = _clamp(box.x2, x1, W)
    y2 = _clamp(box.y2, y1, H)

    original_area = (original_box.x2 - original_box.x1) * (original_box.y2 - original_box.y1)
    adjusted_area = (x2 - x1) * (y2 - y1)
    area_ratio = adjusted_area / original_area if original_area > 0 else 0.0

    adjustment_meta = {
        "adjusted": True,
        "area_ratio": area_ratio,
    }
    return template.model_copy(
        update={
            "top_left_x": x1 / W,
            "top_left_y": y1 / H,
            "width": (x2 - x1) / W,
            "height": (y2 - y1) / H,
            "meta": {**(template.meta or {}), _ADJUSTMENT_META_KEY: adjustment_meta},
        }
    )


def _point_on_segment(p: PixelPoint, a: PixelPoint, b: PixelPoint, tolerance_px: float = 0.0) -> bool:
    """Return True if pixel point ``p`` lies on segment ``a``-``b``.

    With ``tolerance_px > 0`` a point within that many pixels of the segment also counts as "on" it.
    With ``tolerance_px == 0`` an exact collinear-and-within-bounds test is used.
    """
    px, py = p
    ax, ay = a
    bx, by = b
    if tolerance_px > 0.0:
        dx, dy = bx - ax, by - ay
        seg_len_sq = dx * dx + dy * dy
        if seg_len_sq == 0.0:  # Degenerate segment (a == b)
            distance = math.hypot(px - ax, py - ay)
        else:
            t = _clamp(((px - ax) * dx + (py - ay) * dy) / seg_len_sq, 0.0, 1.0)
            distance = math.hypot(px - (ax + t * dx), py - (ay + t * dy))
        return distance <= tolerance_px

    cross = (px - ax) * (by - ay) - (py - ay) * (bx - ax)
    if abs(cross) > _COLLINEAR_EPS_PX:
        return False
    return min(ax, bx) <= px <= max(ax, bx) and min(ay, by) <= py <= max(ay, by)


def _point_in_poly_inclusive(point: PixelPoint, polygon_px: PixelPolygon, tolerance_px: float = 0.0) -> bool:
    n = len(polygon_px)
    if n < 3:
        return False
    px, py = point
    inside = False
    for i in range(n):
        x0, y0 = polygon_px[i]
        x1, y1 = polygon_px[(i + 1) % n]
        if _point_on_segment(point, (x0, y0), (x1, y1), tolerance_px):
            return True
        if (y0 > py) != (y1 > py):
            xinters = x0 + (py - y0) * (x1 - x0) / (y1 - y0)
            if px <= xinters:
                inside = not inside
    return inside


def _corners(box: PixelBbox) -> Tuple[PixelPoint, PixelPoint, PixelPoint, PixelPoint]:
    return (
        (box.x1, box.y1),  # TL
        (box.x2, box.y1),  # TR
        (box.x2, box.y2),  # BR
        (box.x1, box.y2),  # BL
    )


def _corners_states(
    box: PixelBbox, polygon_px: PixelPolygon, tolerance_px: float = 0.0
) -> Tuple[bool, bool, bool, bool]:
    tl, tr, br, bl = _corners(box)
    return (
        _point_in_poly_inclusive(tl, polygon_px, tolerance_px),  # TL
        _point_in_poly_inclusive(tr, polygon_px, tolerance_px),  # TR
        _point_in_poly_inclusive(br, polygon_px, tolerance_px),  # BR
        _point_in_poly_inclusive(bl, polygon_px, tolerance_px),  # BL
    )


def _first_adjacent_pair_inside(box: PixelBbox, polygon_px: PixelPolygon) -> Optional[str]:
    iTL, iTR, iBR, iBL = _corners_states(box, polygon_px)
    if iTL and iTR:
        return "top"
    if iTR and iBR:
        return "right"
    if iBR and iBL:
        return "bottom"
    if iBL and iTL:
        return "left"
    return None


def _valid_bbox(box: PixelBbox) -> bool:
    return box.x2 > box.x1 and box.y2 > box.y1


def _adjust_side_minimal(box: PixelBbox, side: str, polygon_px: PixelPolygon) -> PixelBbox:
    """
    Minimally shrink one side (by whole pixels) so that its pair of adjacent corners is no longer inside.
    RULE: always REDUCE the box (never expand it).
      - top:    move top edge down
      - bottom: move bottom edge up
      - left:   move left edge right
      - right:  move right edge left
    """

    def both_inside_for_side(candidate: PixelBbox) -> bool:
        iTL, iTR, iBR, iBL = _corners_states(candidate, polygon_px)
        return {"top": iTL and iTR, "right": iTR and iBR, "bottom": iBR and iBL, "left": iBL and iTL}[side]

    if not both_inside_for_side(box):
        return box

    # A side only needs to travel across its own extent before the box collapses; further steps are invalid.
    extent = (box.y2 - box.y1) if side in ("top", "bottom") else (box.x2 - box.x1)
    for step in range(1, int(math.ceil(extent)) + 1):
        if side == "top":
            candidate = PixelBbox(box.x1, box.y1 + step, box.x2, box.y2)
        elif side == "bottom":
            candidate = PixelBbox(box.x1, box.y1, box.x2, box.y2 - step)
        elif side == "left":
            candidate = PixelBbox(box.x1 + step, box.y1, box.x2, box.y2)
        elif side == "right":
            candidate = PixelBbox(box.x1, box.y1, box.x2 - step, box.y2)
        else:
            raise ValueError(f"Unknown side: {side}")

        # Keep the box valid and check that the adjacent pair is no longer inside
        if _valid_bbox(candidate) and not both_inside_for_side(candidate):
            return candidate

    return box
