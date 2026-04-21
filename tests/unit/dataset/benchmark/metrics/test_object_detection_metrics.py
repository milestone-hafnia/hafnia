from pathlib import Path

import pytest
from PIL import Image

from hafnia.dataset.benchmark.benchmark import run_inference_on_dataset
from hafnia.dataset.dataset_names import TASK_NAME_PREDICTIONS_POSTFIX
from hafnia.dataset.hafnia_dataset import HafniaDataset
from hafnia.dataset.hafnia_dataset_types import DatasetInfo, Sample, TaskInfo
from hafnia.dataset.primitives import Bbox, Bitmask
from tests import helper_testing
from tests.helper_testing_benchmark import FakeInferenceModel


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
    model_task_names_before = [t.name for t in model.get_model_tasks()]
    dataset_with_predictions = run_inference_on_dataset(dataset=gt_dataset, model=model)
    model_task_names_after = [t.name for t in model.get_model_tasks()]
    assert model_task_names_before == model_task_names_after, (
        f"Model task names should not be modified by run_inference_on_dataset, "
        f"expected {model_task_names_before}, got {model_task_names_after}"
    )
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
