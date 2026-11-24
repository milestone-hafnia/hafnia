from typing import List, Optional

import pytest

from hafnia.dataset.benchmark.benchmark import (
    align_prediction_classes,
    batched,
    run_benchmark,
    run_inference_on_dataset,
)
from hafnia.dataset.hafnia_dataset import HafniaDataset
from hafnia.dataset.hafnia_dataset_types import TaskInfo
from hafnia.dataset.primitives import Bbox, Primitive
from tests import helper_testing
from tests.helper_testing_benchmark import FakeInferenceModel


@pytest.mark.parametrize("dataset_name", helper_testing.MICRO_DATASETS)
def test_benchmark(dataset_name: str):
    path_dataset = helper_testing.get_path_micro_hafnia_dataset(dataset_name=dataset_name, force_update=False)

    gt_dataset = HafniaDataset.from_path(path_dataset)
    model = FakeInferenceModel(fake_model_tasks=gt_dataset.info.tasks)
    task_name_prediction_postfix = "/some_predictions"
    metrics, dataset_predictions = run_benchmark(
        dataset=gt_dataset,
        model=model,
        task_name_prediction_postfix=task_name_prediction_postfix,
    )

    dataset_prediction_tasks = {t.name for t in dataset_predictions.info.tasks}
    assert len(metrics) > 0, "Expected at least one metric to be calculated"
    assert len(dataset_predictions) == len(gt_dataset)
    for model_task in model.get_model_info().tasks:
        assert model_task.name is not None, "Model task names cannot be None"
        prediction_task_name = model_task.name + task_name_prediction_postfix
        assert prediction_task_name in dataset_prediction_tasks


EXTRA_CLASS_NAME = "extra_class"


class ExtraClassInferenceModel(FakeInferenceModel):
    """A 'FakeInferenceModel' that also predicts one box of a class the dataset doesn't have on each image."""

    def predict(self, image, sample_dict=None) -> List[Primitive]:
        predictions = super().predict(image, sample_dict=sample_dict)
        if predictions:
            extra_prediction = predictions[0].model_copy(deep=True)
            extra_prediction.class_name = EXTRA_CLASS_NAME
            extra_prediction.class_idx = 0
            predictions.append(extra_prediction)
        return predictions


def detection_model_task(class_names: List[str]) -> TaskInfo:
    return TaskInfo.from_class_names(primitive=Bbox, class_names=class_names, name="object_detection")


def test_run_benchmark_auto_align_classes():
    """A model with an extra class and another class order: aligned, it scores as the ground truth itself."""
    gt_dataset = helper_testing.get_micro_hafnia_dataset("micro-coco-2017")
    gt_class_names = gt_dataset.info.get_task_by_name("object_detection").get_class_names() or []
    model_class_names = [EXTRA_CLASS_NAME, *reversed(gt_class_names)]
    model = ExtraClassInferenceModel(fake_model_tasks=[detection_model_task(model_class_names)])

    with pytest.raises(ValueError, match="auto_align_classes=True"):
        run_benchmark(dataset=gt_dataset, model=model)
    metrics, dataset_predictions = run_benchmark(dataset=gt_dataset, model=model, auto_align_classes=True)

    prediction_task = dataset_predictions.info.get_task_by_name("object_detection/predictions")
    assert prediction_task.get_class_names() == gt_class_names
    predicted_class_names = {
        bbox["class_name"] for bboxes in dataset_predictions.samples["bboxes"].to_list() for bbox in bboxes
    }
    assert EXTRA_CLASS_NAME not in predicted_class_names

    reference_model = FakeInferenceModel(fake_model_tasks=[detection_model_task(gt_class_names)])
    reference_metrics, reference_predictions = run_benchmark(dataset=gt_dataset, model=reference_model)
    assert metrics == pytest.approx(reference_metrics)
    assert align_prediction_classes(reference_predictions) is reference_predictions, "Matching classes are unchanged"


def test_align_prediction_classes_raises_when_label_spaces_differ():
    gt_dataset = helper_testing.get_micro_hafnia_dataset("micro-coco-2017")
    gt_class_names = gt_dataset.info.get_task_by_name("object_detection").get_class_names() or []
    model = FakeInferenceModel(fake_model_tasks=[detection_model_task(gt_class_names[1:])])

    dataset_predictions = run_inference_on_dataset(dataset=gt_dataset, model=model)
    with pytest.raises(ValueError, match=f"has no classes \\['{gt_class_names[0]}'\\]"):
        align_prediction_classes(dataset_predictions)


class BatchRecordingInferenceModel(FakeInferenceModel):
    """A 'FakeInferenceModel' that records the size of each batch passed to 'predict_batch'."""

    def __init__(self, fake_model_tasks):
        super().__init__(fake_model_tasks=fake_model_tasks)
        self.batch_sizes: List[int] = []

    def predict_batch(self, images, sample_dicts=None) -> List[List[Primitive]]:
        self.batch_sizes.append(len(images))
        return super().predict_batch(images, sample_dicts=sample_dicts)


@pytest.mark.parametrize(
    "batch_size, expected_batch_sizes",
    [(1, [1, 1, 1]), (2, [2, 1]), (3, [3]), (10, [3])],
)
def test_run_inference_on_dataset_uses_predict_batch(batch_size: int, expected_batch_sizes: List[int]):
    """Inference is routed through 'predict_batch' and the dataset is split into batches of 'batch_size'."""
    gt_dataset = helper_testing.get_micro_hafnia_dataset("micro-tiny-dataset")
    assert len(gt_dataset) == 3, "The expected batch sizes below assume a 3-sample dataset"
    model = BatchRecordingInferenceModel(fake_model_tasks=gt_dataset.info.tasks)

    dataset_predictions = run_inference_on_dataset(dataset=gt_dataset, model=model, batch_size=batch_size)

    assert model.batch_sizes == expected_batch_sizes
    assert len(dataset_predictions) == len(gt_dataset), "Every sample should be present in the prediction dataset"


def test_run_inference_on_dataset_batching_does_not_change_predictions():
    """Batching only changes how 'predict' is called - not the resulting prediction dataset."""
    gt_dataset = helper_testing.get_micro_hafnia_dataset("micro-tiny-dataset")

    predictions_unbatched = run_inference_on_dataset(
        dataset=gt_dataset, model=FakeInferenceModel(fake_model_tasks=gt_dataset.info.tasks), batch_size=1
    )
    predictions_batched = run_inference_on_dataset(
        dataset=gt_dataset, model=FakeInferenceModel(fake_model_tasks=gt_dataset.info.tasks), batch_size=3
    )

    assert predictions_batched.samples.equals(predictions_unbatched.samples)
    assert predictions_batched.info == predictions_unbatched.info


def test_run_inference_on_dataset_raises_on_invalid_batch_size():
    gt_dataset = helper_testing.get_micro_hafnia_dataset("micro-tiny-dataset")
    model = FakeInferenceModel(fake_model_tasks=gt_dataset.info.tasks)

    with pytest.raises(ValueError, match="'batch_size' must be at least 1"):
        run_inference_on_dataset(dataset=gt_dataset, model=model, batch_size=0)


def test_run_inference_on_dataset_raises_when_predict_batch_returns_wrong_length():
    """A bad 'predict_batch' override must not silently misalign predictions with samples."""

    class BrokenBatchModel(FakeInferenceModel):
        def predict_batch(self, images, sample_dicts: Optional[List[dict]] = None) -> List[List[Primitive]]:
            return [[]]  # One list of primitives regardless of the number of images

    gt_dataset = helper_testing.get_micro_hafnia_dataset("micro-tiny-dataset")
    model = BrokenBatchModel(fake_model_tasks=gt_dataset.info.tasks)

    with pytest.raises(ValueError, match="returned predictions for 1 images, but it was given 2 images"):
        run_inference_on_dataset(dataset=gt_dataset, model=model, batch_size=2)


def test_batched():
    assert list(batched(range(5), batch_size=2)) == [[0, 1], [2, 3], [4]]
    assert list(batched(range(4), batch_size=4)) == [[0, 1, 2, 3]]
    assert list(batched([], batch_size=3)) == []
