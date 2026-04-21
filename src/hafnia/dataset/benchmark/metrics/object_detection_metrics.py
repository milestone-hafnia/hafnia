from __future__ import annotations

import io
from contextlib import redirect_stdout
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Type

import numpy as np

from hafnia.dataset.primitives import Bbox, Bitmask, Primitive
from hafnia.utils import progress_bar

if TYPE_CHECKING:
    from hafnia.dataset.hafnia_dataset import HafniaDataset


@dataclass
class MapMetrics:
    """Standard COCO mAP metrics (12 metrics from COCOeval.summarize)."""

    ap: float  # AP @ IoU=0.50:0.95 (primary metric)
    ap50: float  # AP @ IoU=0.50
    ap75: float  # AP @ IoU=0.75
    ap_s: float  # AP for small objects  (area < 32² px)
    ap_m: float  # AP for medium objects (32² < area < 96² px)
    ap_l: float  # AP for large objects  (area > 96² px)
    ar1: float  # AR with max 1 detection per image
    ar10: float  # AR with max 10 detections per image
    ar100: float  # AR with max 100 detections per image
    ar_s: float  # AR for small objects
    ar_m: float  # AR for medium objects
    ar_l: float  # AR for large objects

    @staticmethod
    def from_coco_stats(stats: List[np.float64]) -> "MapMetrics":
        stats_as_native_float = [s.item() for s in stats]  # Converts from numpy.float64 to native float
        return MapMetrics(
            ap=stats_as_native_float[0],
            ap50=stats_as_native_float[1],
            ap75=stats_as_native_float[2],
            ap_s=stats_as_native_float[3],
            ap_m=stats_as_native_float[4],
            ap_l=stats_as_native_float[5],
            ar1=stats_as_native_float[6],
            ar10=stats_as_native_float[7],
            ar100=stats_as_native_float[8],
            ar_s=stats_as_native_float[9],
            ar_m=stats_as_native_float[10],
            ar_l=stats_as_native_float[11],
        )

    def as_dict(self, upper: bool = False) -> Dict[str, float]:
        """Return metrics as a dictionary."""
        metrics_dict = asdict(self)
        if upper:
            metrics_dict = {key.upper(): value for key, value in metrics_dict.items()}
        return metrics_dict


def _build_coco_data(
    dataset: "HafniaDataset",
    primitive: Type[Primitive],
    task_name_predictions: str,
    task_name_ground_truth: str,
    category_mapping: Dict[str, int],
    categories: List[Dict[str, Any]],
) -> tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Iterate samples to build COCO GT dict and predictions list."""
    images: List[Dict[str, Any]] = []
    annotations: List[Dict[str, Any]] = []
    predictions: List[Dict[str, Any]] = []
    ann_id = 1  # pycocotools treats id=0 as "no match" in boolean logic; IDs must start at 1
    column_name = primitive.column_name()  # "bboxes" or "bitmasks"

    for row in progress_bar(
        dataset.samples.iter_rows(named=True),
        description="Convert to 'pycocotools' format",
        total=len(dataset),
    ):
        sample_index = int(row["sample_index"])
        img_height = int(row["height"])
        img_width = int(row["width"])

        images.append(
            {
                "id": sample_index,
                "width": img_width,
                "height": img_height,
                "file_name": row.get("file_path") or "",
            }
        )

        for ann in row.get(column_name) or []:
            if ann is None:
                continue

            ann_task_name = ann.get("task_name")
            class_name = ann.get("class_name")
            if class_name not in category_mapping:
                continue
            cat_id = category_mapping[class_name]

            if ann_task_name == task_name_ground_truth:
                iscrowd = int((ann.get("meta") or {}).get("iscrowd") or 0)

                if primitive is Bbox:
                    bbox = [
                        ann["top_left_x"] * img_width,
                        ann["top_left_y"] * img_height,
                        ann["width"] * img_width,
                        ann["height"] * img_height,
                    ]
                    area = float(ann["height"] * ann["width"] * img_height * img_width)
                    segmentation: Any = []
                else:  # Bitmask
                    bbox = [
                        float(ann["left"]),
                        float(ann["top"]),
                        float(ann["width"]),
                        float(ann["height"]),
                    ]
                    area = float((ann.get("area") or 0.0) * img_height * img_width)
                    rle_counts = ann["rle_string"]
                    if isinstance(rle_counts, str):
                        rle_counts = rle_counts.encode("utf-8")
                    segmentation = {"counts": rle_counts, "size": [img_height, img_width]}

                annotations.append(
                    {
                        "id": ann_id,
                        "image_id": sample_index,
                        "category_id": cat_id,
                        "bbox": bbox,
                        "area": area,
                        "segmentation": segmentation,
                        "iscrowd": iscrowd,
                    }
                )
                ann_id += 1

            elif ann_task_name == task_name_predictions:
                score = float(ann.get("confidence") or 1.0)

                if primitive is Bbox:
                    pred: Dict[str, Any] = {
                        "image_id": sample_index,
                        "category_id": cat_id,
                        "bbox": [
                            ann["top_left_x"] * img_width,
                            ann["top_left_y"] * img_height,
                            ann["width"] * img_width,
                            ann["height"] * img_height,
                        ],
                        "score": score,
                        "area": float(ann["height"] * ann["width"] * img_height * img_width),
                    }
                else:  # Bitmask
                    rle_counts = ann["rle_string"]
                    if isinstance(rle_counts, str):
                        rle_counts = rle_counts.encode("utf-8")
                    pred = {
                        "image_id": sample_index,
                        "category_id": cat_id,
                        "segmentation": {"counts": rle_counts, "size": [img_height, img_width]},
                        "score": score,
                        "area": float((ann.get("area") or 0.0) * img_height * img_width),
                    }

                predictions.append(pred)

    gt_dict = {"images": images, "annotations": annotations, "categories": categories}
    return gt_dict, predictions


def calculate_map(
    dataset: "HafniaDataset",
    task_name_predictions: str,
    task_name_ground_truth: str,
) -> MapMetrics:
    """Calculate mean average precision (mAP) using the COCO evaluation protocol.

    Both ground-truth and prediction annotations must be stored in the same
    dataset, distinguished by their ``task_name`` field.  The primitive type
    (Bbox → ``"bbox"`` eval, Bitmask → ``"segm"`` eval) is inferred from the
    dataset's :class:`~hafnia.dataset.hafnia_dataset_types.TaskInfo` that
    matches *task_name_ground_truth*.

    Typical usage::

        metrics = dataset_with_predictions.calculate_map(
            task_name_predictions="predictions",
            task_name_ground_truth=Bbox.default_task_name(),
        )

    Args:
        dataset: Dataset containing both ground-truth and prediction annotations.
        task_name_predictions: Task name that identifies prediction annotations.
        task_name_ground_truth: Task name that identifies ground-truth annotations.

    Returns:
        A :class:`MapMetrics` dataclass with the 12 standard COCO metrics.
    """
    # import faster_coco_eval
    # faster_coco_eval.init_as_pycocotools()

    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    task_info = dataset.info.get_task_by_name(task_name_ground_truth)
    primitive = task_info.primitive

    if primitive is Bbox:
        eval_type = "bbox"
    elif primitive is Bitmask:
        eval_type = "segm"
    else:
        raise ValueError(
            f"Unsupported primitive type for mAP calculation: {primitive}. "
            "Only Bbox ('bbox') and Bitmask ('segm') are supported."
        )

    class_names = task_info.get_class_names()
    if class_names is None:
        raise ValueError(
            f"Ground-truth task '{task_name_ground_truth}' does not define any class names. "
            "mAP calculation requires at least one class name to build COCO categories."
        )
    id_offset = 1  # COCO doesn't work nicely with 0-indexing so we start at 1.
    categories = [
        {"id": i, "name": name, "supercategory": "object"} for i, name in enumerate(class_names, start=id_offset)
    ]
    category_mapping = {str(cat["name"]): cat["id"] for cat in categories}

    gt_dict, pred_list = _build_coco_data(
        dataset=dataset,
        primitive=primitive,
        task_name_predictions=task_name_predictions,
        task_name_ground_truth=task_name_ground_truth,
        category_mapping=category_mapping,  # type: ignore[arg-type]
        categories=categories,
    )
    if not pred_list:
        raise ValueError(
            f"No predictions found for task '{task_name_predictions}'. "
            "Ensure predictions are stored in the dataset with the correct task_name."
        )

    gt_dict.update({"info": {}})
    gt_coco = COCO()
    gt_coco.dataset = gt_dict
    with redirect_stdout(io.StringIO()):
        gt_coco.createIndex()

    pred_coco = gt_coco.loadRes(pred_list)

    coco_eval = COCOeval(gt_coco, pred_coco, eval_type)
    with redirect_stdout(io.StringIO()):  # Suppress COCOeval print output
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
    return MapMetrics.from_coco_stats(list(coco_eval.stats))
