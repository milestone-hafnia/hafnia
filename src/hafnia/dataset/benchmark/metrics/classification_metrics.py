from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Dict, List

import polars as pl

from hafnia.dataset.dataset_names import PrimitiveField, SampleField
from hafnia.dataset.primitives import Classification

if TYPE_CHECKING:
    from hafnia.dataset.hafnia_dataset import HafniaDataset


@dataclass
class PerClassClassificationMetrics:
    accuracy: float
    precision: float
    recall: float
    f1: float
    support: int

    @staticmethod
    def empty() -> "PerClassClassificationMetrics":
        return PerClassClassificationMetrics(accuracy=0.0, precision=0.0, recall=0.0, f1=0.0, support=0)

    @staticmethod
    def from_confusion(TP: int, FP: int, FN: int, TN: int) -> "PerClassClassificationMetrics":
        accuracy = (TP + TN) / (TP + FP + FN + TN) if (TP + FP + FN + TN) > 0 else 0.0
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        f1 = 2 * TP / (2 * TP + FP + FN) if (2 * TP + FP + FN) > 0 else 0.0
        support = TP + FN
        return PerClassClassificationMetrics(
            accuracy=accuracy, precision=precision, recall=recall, f1=f1, support=support
        )

    def as_dict(self) -> Dict[str, float]:
        return asdict(self)


@dataclass
class ClassificationMetrics:
    """Classification metrics for a single-label classification task.

    Metrics are computed per-class from a confusion matrix and aggregated
    using macro and weighted averaging.
    """

    accuracy: float
    precision_macro: float
    recall_macro: float
    f1_macro: float
    precision_weighted: float
    recall_weighted: float
    f1_weighted: float
    n_gt_missing: int
    n_pred_missing: int
    per_class: Dict[str, PerClassClassificationMetrics]

    def as_dict(self) -> Dict[str, float]:
        """Return flat scalar metrics dict (excludes per_class and confusion_matrix)."""
        d = asdict(self)
        per_class = d.pop("per_class")
        per_class_flatten = {
            f"{class_name}.{metric_name}": metric_value
            for class_name, metrics in per_class.items()
            for metric_name, metric_value in metrics.items()
        }
        d.update(per_class_flatten)
        return d


def _extract_class_names_for_task(
    samples: pl.DataFrame,
    task_name: str,
) -> pl.Series:
    """Extract the first class_name per sample for a given task from the classifications column.

    Returns a String series with one entry per sample row (null when no match).
    """
    col = SampleField.CLASSIFICATIONS
    return samples.select(
        pl.col(col)
        .list.eval(
            pl.when(pl.element().struct.field(PrimitiveField.TASK_NAME) == task_name)
            .then(pl.element().struct.field(PrimitiveField.CLASS_NAME))
            .otherwise(pl.lit(None, dtype=pl.String))
            .drop_nulls()
            .first()
        )
        .list.first()
        .alias("class_name")
    ).to_series()


def _compute_metrics_from_pairs(
    gt_classes: pl.Series,
    pred_classes: pl.Series,
    class_names: List[str],
) -> ClassificationMetrics:
    """Compute classification metrics from paired ground-truth / prediction class name series.

    Null handling:
      - GT null:   sample has no ground truth — dropped (counted as ``n_gt_missing``).
      - Pred null:  model failed to predict — treated as an error that penalizes
        recall / F1 for the ground-truth class (counted as ``n_pred_missing``).
    """
    pairs = pl.DataFrame({"gt": gt_classes, "pred": pred_classes})

    # Samples with no ground truth — drop, count
    n_gt_missing = int(pairs.select(pl.col("gt").is_null().sum()).item())
    pairs = pairs.filter(pl.col("gt").is_not_null())

    # Samples with GT but no prediction — counted but kept in `pairs` so they
    # contribute an FN to their GT class. (FN is derived as support - TP below,
    # which naturally includes null-pred rows without special-casing.)
    n_pred_missing = int(pairs.select(pl.col("pred").is_null().sum()).item())

    total_evaluated = len(pairs)

    if total_evaluated == 0:
        return ClassificationMetrics(
            accuracy=0.0,
            precision_macro=0.0,
            recall_macro=0.0,
            f1_macro=0.0,
            precision_weighted=0.0,
            recall_weighted=0.0,
            f1_weighted=0.0,
            n_gt_missing=n_gt_missing,
            n_pred_missing=n_pred_missing,
            per_class={name: PerClassClassificationMetrics.empty() for name in class_names},
        )

    # Confusion counts in long form: one row per observed (gt, pred) pair.
    # A null prediction forms its own group; it never matches any real class in
    # the TP/FP filters below, so it neither contributes a TP nor an FP to any
    # real class — which is exactly what we want.
    cm_long = pairs.group_by(["gt", "pred"]).agg(pl.len().alias("count"))

    # Per-class TP / FP / FN / TN.
    #   support[c] = rows where gt == c
    #   TP[c]      = rows where gt == c AND pred == c
    #   FP[c]      = rows where gt != c AND pred == c
    #   FN[c]      = support[c] - TP[c]
    #   TN[c]      = total_evaluated - TP - FP - FN
    #
    # FN is derived from support (not `pred != c`) because polars uses
    # three-valued logic: `null != c` evaluates to null, which `.filter()`
    # treats as false and drops — that would silently miss null-pred rows.
    per_class: Dict[str, PerClassClassificationMetrics] = {}
    total_correct = 0
    for c in class_names:
        support = int(cm_long.filter(pl.col("gt") == c)["count"].sum())
        tp = int(cm_long.filter((pl.col("gt") == c) & (pl.col("pred") == c))["count"].sum())
        fp = int(cm_long.filter((pl.col("gt") != c) & (pl.col("pred") == c))["count"].sum())
        fn = support - tp
        tn = total_evaluated - tp - fp - fn
        per_class[c] = PerClassClassificationMetrics.from_confusion(TP=tp, FP=fp, FN=fn, TN=tn)
        total_correct += tp

    # Overall accuracy. In a single-label problem each sample is a TP for at
    # most one class, so the sum of per-class TPs equals the number of correct
    # predictions.
    accuracy = total_correct / total_evaluated

    # Macro / weighted averages over real classes.
    n_cls = len(class_names)
    precisions = [per_class[c].precision for c in class_names]
    recalls = [per_class[c].recall for c in class_names]
    f1s = [per_class[c].f1 for c in class_names]
    supports = [per_class[c].support for c in class_names]

    precision_macro = sum(precisions) / n_cls
    recall_macro = sum(recalls) / n_cls
    f1_macro = sum(f1s) / n_cls

    total_support = sum(supports)
    if total_support > 0:
        precision_weighted = sum(p * s for p, s in zip(precisions, supports)) / total_support
        recall_weighted = sum(r * s for r, s in zip(recalls, supports)) / total_support
        f1_weighted = sum(f * s for f, s in zip(f1s, supports)) / total_support
    else:
        precision_weighted = 0.0
        recall_weighted = 0.0
        f1_weighted = 0.0

    return ClassificationMetrics(
        accuracy=accuracy,
        precision_macro=precision_macro,
        recall_macro=recall_macro,
        f1_macro=f1_macro,
        precision_weighted=precision_weighted,
        recall_weighted=recall_weighted,
        f1_weighted=f1_weighted,
        n_gt_missing=n_gt_missing,
        n_pred_missing=n_pred_missing,
        per_class=per_class,
    )


def calculate_classification_metrics(
    dataset: "HafniaDataset",
    task_name_predictions: str,
    task_name_ground_truth: str,
) -> ClassificationMetrics:
    """Calculate classification metrics comparing predictions to ground truth.

    Both ground-truth and prediction annotations must be stored in the same
    dataset as Classification primitives, distinguished by their ``task_name``.

    Unlike most classifications metrics calculators, this handles both missing
    predictions and missing ground truth:
      - Missing GT:   sample is not evaluated (dropped), counted in ``n_gt_missing``.
      - Missing pred: sample is evaluated as incorrect (FN for its GT class), counted in ``n_pred_missing``.


    Typical usage::

        metrics = dataset.calculate_classification_metrics(
            task_name_predictions="image_classification/predictions",
            task_name_ground_truth=Classification.default_task_name(),
        )

    Args:
        dataset: Dataset containing both ground-truth and prediction classifications.
        task_name_predictions: Task name identifying prediction classifications.
        task_name_ground_truth: Task name identifying ground-truth classifications.

    Returns:
        A :class:`ClassificationMetrics` dataclass.
    """
    if SampleField.CLASSIFICATIONS not in dataset.samples.columns:
        raise ValueError(
            f"Dataset does not contain a '{SampleField.CLASSIFICATIONS}' column. Cannot compute classification metrics."
        )

    ground_truth_task = dataset.info.get_task_by_name(task_name_ground_truth)
    if ground_truth_task.primitive is not Classification:
        raise ValueError(
            f"Ground-truth task '{task_name_ground_truth}' has primitive "
            f"'{ground_truth_task.primitive.__name__}', expected 'Classification'."
        )
    class_names = ground_truth_task.get_class_names() or []

    if not class_names:
        raise ValueError(
            f"No class names found for ground-truth task '{task_name_ground_truth}'. "
            "Provide class_names explicitly or ensure the task has classes defined."
        )

    prediction_task = dataset.info.get_task_by_name(task_name_predictions)
    if prediction_task.primitive is not Classification:
        raise ValueError(
            f"Prediction task '{task_name_predictions}' has primitive "
            f"'{prediction_task.primitive.__name__}', expected 'Classification'."
        )
    prediction_class_names = prediction_task.get_class_names() or []
    if prediction_class_names != class_names:  # Compare lists directly to also ensure order matches
        raise ValueError(
            f"Class names between ground-truth and prediction tasks do not match "
            f"Prediction task '{task_name_predictions}' has class names "
            f"'{prediction_class_names}', which do not match ground-truth class names "
            f"in task '{task_name_ground_truth}': '{class_names}'. "
        )

    gt_classes = _extract_class_names_for_task(dataset.samples, task_name_ground_truth)
    pred_classes = _extract_class_names_for_task(dataset.samples, task_name_predictions)

    return _compute_metrics_from_pairs(gt_classes, pred_classes, class_names)
