from pathlib import Path

import pytest
from PIL import Image

from hafnia.benchmark.benchmark import run_inference_on_dataset
from hafnia.dataset.dataset_names import TASK_NAME_PREDICTIONS_POSTFIX
from hafnia.dataset.hafnia_dataset import HafniaDataset
from hafnia.dataset.hafnia_dataset_types import DatasetInfo, Sample, TaskInfo
from hafnia.dataset.primitives import Bbox, Bitmask
from tests import helper_testing
from tests.helper_testing_benchmark import FakeInferenceModel

# def test_calculate_map_perfect_predictions():
#     """Test that calculate_map returns AP=1.0 when predictions are perfect copies of GT.

#     Uses a fully controlled synthetic dataset to avoid edge-case artifacts present
#     in real micro datasets:
#       - AP < 1.0 on per-area sub-metrics (small/medium/large) due to pycocotools'
#         global score-sorted matching: when many same-score predictions exist in one
#         image, ties are broken by list order and predictions of one area bucket can
#         steal GT matches from another bucket, artificially lowering per-area AP.
#       - AR@maxDets=1 < 1.0 by design: with more than one GT object per image this
#         metric is always below 1.0 regardless of prediction quality.
#       - GT annotation ids starting at 0 cause pycocotools to treat the first matched
#         detection as a false positive (dtMatches stores the GT id and 0 is falsy).

#     To avoid these issues this test uses:
#       - Multiple images, each with a single well-separated "large" GT box per class,
#         so no same-score ties can cause cross-area stealing.
#       - Only the area=all and IoU=0.50 metrics are asserted.
#     """
#     # Three images, each 500x500, one large box per class, no overlap.
#     class_names = ["cat", "dog"]
#     samples = []
#     for i in range(3):
#         bboxes = [
#             Bbox(top_left_x=0.0, top_left_y=0.0, width=0.4, height=0.4, class_name="cat"),
#             Bbox(top_left_x=0.6, top_left_y=0.6, width=0.4, height=0.4, class_name="dog"),
#         ]
#         samples.append(
#             Sample(file_path=f"/tmp/synthetic_img_{i}.jpg", split="train", height=500, width=500, bboxes=bboxes)
#         )

#     info = DatasetInfo(
#         dataset_name="test_map",
#         tasks=[TaskInfo.from_class_names(primitive=Bbox, class_names=class_names)],
#     )
#     gt_dataset = HafniaDataset.from_samples_list(samples, info=info)

#     updated_samples = []
#     for dict_sample in gt_dataset:
#         sample = Sample(**dict_sample)
#         predictions = []
#         for ann in sample.get_primitives(primitive_types=[Bbox]):
#             pred = ann.model_copy(deep=True)
#             pred.task_name = "predictions"
#             pred.confidence = 0.9
#             pred.ground_truth = False
#             predictions.append(pred)
#         sample.append_primitives(predictions)
#         updated_samples.append(sample)

#     dataset_with_predictions = HafniaDataset.from_samples_list(updated_samples, info=info)
#     metrics = dataset_with_predictions.calculate_map(
#         task_name_predictions="predictions",
#         task_name_ground_truth=Bbox.default_task_name(),
#     )

#     assert metrics.ap == pytest.approx(1.0), f"Expected AP=1.0, got {metrics.ap}"
#     assert metrics.ap50 == pytest.approx(1.0), f"Expected AP50=1.0, got {metrics.ap50}"
#     assert metrics.ar100 == pytest.approx(1.0), f"Expected AR100=1.0, got {metrics.ar100}"


# @pytest.mark.parametrize("dataset_name", helper_testing.MICRO_DATASETS)
# def test_calculate_map(dataset_name: str, tmp_path: Path):
#     path_dataset = helper_testing.get_path_micro_hafnia_dataset(dataset_name=dataset_name, force_update=False)

#     gt_dataset = HafniaDataset.from_path(path_dataset)
#     gt_dataset.print_class_distribution()

#     def fake_model_predictions(sample: Sample, primitive_type: Type[Primitive]) -> List[Primitive]:
#         gt_annotations = sample.get_primitives(primitive_types=[primitive_type])
#         predictions = []
#         for ann in gt_annotations:
#             if (ann.meta or {}).get("iscrowd"):
#                 print("SKIP CROWD ANNOTATION")
#                 continue  # skip crowd annotations
#             pred_ann: Primitive = ann.model_copy(deep=True)
#             pred_ann.task_name = "predictions"
#             pred_ann.confidence = 0.9
#             pred_ann.ground_truth = False
#             predictions.append(pred_ann)
#         return predictions

#     coco_tasks = Bbox, Bitmask
#     available_coco_metric_tasks = {t.primitive for t in gt_dataset.info.tasks if t.primitive in coco_tasks}
#     assert len(available_coco_metric_tasks) > 0, (
#         f"Expected exactly one of {coco_tasks} in dataset tasks, found: {available_coco_metric_tasks}"
#     )

#     for model_primitive in available_coco_metric_tasks:
#         # ann_count = 0
#         updated_samples = []
#         for dict_sample in gt_dataset:
#             sample = Sample(**dict_sample)
#             predictions = fake_model_predictions(sample, primitive_type=model_primitive)
#             sample.append_primitives(predictions)
#             updated_samples.append(sample)

#         dataset_with_pred = HafniaDataset.from_samples_list(updated_samples, info=gt_dataset.info)

#         n_samples = None  # For testing speed, you may increase this
#         if n_samples is not None:
#             dataset_with_pred = dataset_with_pred.select_samples(n_samples=n_samples, with_replacement=True, seed=42)
#             dataset_with_pred.samples = table_transformations.add_sample_index(dataset_with_pred.samples)

#         metrics = dataset_with_pred.calculate_map(
#             task_name_predictions="predictions",
#             task_name_ground_truth=model_primitive.default_task_name(),
#         )
#         assert metrics.ap == pytest.approx(1.0), (
#             f"Expected AP=1.0, got {metrics.ap}. Ground Truth is directly used as predictions, so AP should be perfect."
#         )

#         assert n_samples is None, "Remember to set 'n_samples = None' before you commit your work!"


def test_calculate_map_perfect_predictions(tmp_path: Path):
    class_names = ["cat", "dog"]
    # run_inference_on_dataset reads each image from disk, so we materialize
    # small black JPEGs in a temporary directory rather than referencing
    # non-existent paths.
    samples = []
    for i in range(3):
        image_path = tmp_path / f"synthetic_img_{i}.jpg"
        Image.new("RGB", (500, 500), color=(0, 0, 0)).save(image_path)
        bboxes = [
            Bbox(top_left_x=0.0, top_left_y=0.0, width=0.4, height=0.4, class_name="cat"),
            Bbox(top_left_x=0.6, top_left_y=0.6, width=0.4, height=0.4, class_name="dog"),
        ]
        samples.append(Sample(file_path=str(image_path), split="train", height=500, width=500, bboxes=bboxes))

    info = DatasetInfo(
        dataset_name="test_map",
        tasks=[TaskInfo.from_class_names(primitive=Bbox, class_names=class_names)],
    )
    gt_dataset = HafniaDataset.from_samples_list(samples, info=info)

    model = FakeInferenceModel(fake_model_tasks=gt_dataset.info.tasks)
    dataset_with_predictions = run_inference_on_dataset(dataset=gt_dataset, model=model)

    gt_task_name = Bbox.default_task_name()
    pred_task_name = f"{gt_task_name}{TASK_NAME_PREDICTIONS_POSTFIX}"
    metrics = dataset_with_predictions.calculate_map(
        task_name_predictions=pred_task_name,
        task_name_ground_truth=gt_task_name,
    )

    assert metrics.ap == pytest.approx(1.0), f"Expected AP=1.0, got {metrics.ap}"
    assert metrics.ap50 == pytest.approx(1.0), f"Expected AP50=1.0, got {metrics.ap50}"
    assert metrics.ar100 == pytest.approx(1.0), f"Expected AR100=1.0, got {metrics.ar100}"


@pytest.mark.parametrize("dataset_name", helper_testing.MICRO_DATASETS)
def test_calculate_map(dataset_name: str):
    path_dataset = helper_testing.get_path_micro_hafnia_dataset(dataset_name=dataset_name, force_update=False)
    gt_dataset = HafniaDataset.from_path(path_dataset)

    coco_primitives = (Bbox, Bitmask)
    model_tasks = [t for t in gt_dataset.info.tasks if t.primitive in coco_primitives]
    assert len(model_tasks) > 0, (
        f"Expected at least one task with primitive in {coco_primitives} in dataset tasks, "
        f"found: {[t.primitive for t in gt_dataset.info.tasks]}"
    )

    for model_task in model_tasks:
        if model_task.name is None:
            raise ValueError(f"Model task names cannot be None, found in dataset '{dataset_name}'")
        model = FakeInferenceModel(fake_model_tasks=[model_task])
        dataset_with_pred = run_inference_on_dataset(dataset=gt_dataset, model=model)

        pred_task_name = f"{model_task.name}{TASK_NAME_PREDICTIONS_POSTFIX}"
        metrics = dataset_with_pred.calculate_map(
            task_name_predictions=pred_task_name,
            task_name_ground_truth=model_task.name,
        )
        assert metrics.ap == pytest.approx(1.0), (
            f"Expected AP=1.0 for task '{model_task.name}' on dataset '{dataset_name}', "
            f"got {metrics.ap}. Ground Truth is directly used as predictions, so AP should be perfect."
        )
