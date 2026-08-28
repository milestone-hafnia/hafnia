from __future__ import annotations

import itertools
import math
from typing import Callable, Dict, Iterable, Iterator, List, Optional, Tuple, Union

from hafnia.dataset.benchmark.inference_model import InferenceModel
from hafnia.dataset.benchmark.metrics_calculator import MetricsCalculator, metric_calculations
from hafnia.dataset.dataset_names import TASK_NAME_PREDICTIONS_POSTFIX
from hafnia.dataset.dataset_recipe.recipe_types import RecipeTransform
from hafnia.dataset.hafnia_dataset import HafniaDataset
from hafnia.dataset.hafnia_dataset_types import Sample
from hafnia.log import user_logger
from hafnia.utils import progress_bar


def run_benchmark(
    dataset: HafniaDataset,
    model: InferenceModel,
    task_name_prediction_postfix: str = TASK_NAME_PREDICTIONS_POSTFIX,
    metric_calculators: Optional[Dict[str, Union[MetricsCalculator, Callable]]] = None,
    recipe_transforms: Optional[List[RecipeTransform]] = None,
    batch_size: int = 1,
) -> Tuple[dict[str, float], HafniaDataset]:
    dataset_predictions = run_inference_on_dataset(
        dataset=dataset,
        model=model,
        task_name_prediction_postfix=task_name_prediction_postfix,
        batch_size=batch_size,
    )

    recipe_transforms = recipe_transforms or []
    for transform in recipe_transforms:
        dataset_predictions = transform.build(dataset_predictions)

    metrics = metric_calculations(
        prediction_dataset=dataset_predictions,
        metric_calculators=metric_calculators,
        prediction_task_name_postfix=task_name_prediction_postfix,
    )
    return metrics, dataset_predictions


def batched(iterable: Iterable, batch_size: int) -> Iterator[list]:
    """Yield consecutive lists of at most 'batch_size' items from 'iterable'."""
    iterator = iter(iterable)
    while batch := list(itertools.islice(iterator, batch_size)):
        yield batch


def run_inference_on_dataset(
    dataset: HafniaDataset,
    model: InferenceModel,
    task_name_prediction_postfix: str = TASK_NAME_PREDICTIONS_POSTFIX,
    batch_size: int = 1,
) -> HafniaDataset:
    if batch_size < 1:
        raise ValueError(f"'batch_size' must be at least 1, but got {batch_size}.")

    model_tasks = [m.model_copy() for m in model.get_model_info().tasks]

    new_task_names = [f"{task.name}{task_name_prediction_postfix}" for task in model_tasks]
    user_logger.info(
        f"Running inference on dataset '{dataset.info.dataset_name}'\n"
        f"- Number of samples: {len(dataset)}\n"
        f"- Model tasks: {[task.name for task in model_tasks]}\n"
        f"- Predictions will be appended to the dataset with new task names:\n"
        f"- Prediction task names: {new_task_names}"
    )

    for model_task in model_tasks:
        model_task.name = f"{model_task.name}{task_name_prediction_postfix}"

    prediction_samples = []
    n_batches = math.ceil(len(dataset) / batch_size)
    batches = batched(dataset, batch_size=batch_size)
    for dict_samples in progress_bar(batches, total=n_batches, description="Running inference on dataset"):
        samples = [Sample(**dict_sample) for dict_sample in dict_samples]
        images = [sample.read_image() for sample in samples]

        batch_predictions = model.predict_batch(images, sample_dicts=dict_samples)
        if len(batch_predictions) != len(samples):
            raise ValueError(
                f"'{type(model).__name__}.predict_batch' returned predictions for {len(batch_predictions)} images, "
                f"but it was given {len(samples)} images. It should return one list of primitives per image."
            )

        for sample, predictions in zip(samples, batch_predictions):
            for prediction in predictions:
                prediction.task_name = f"{prediction.task_name}{task_name_prediction_postfix}"
            sample.append_primitives(predictions)
            prediction_samples.append(sample)

    prediction_dataset_info = dataset.info.model_copy(deep=True)
    prediction_dataset_info.tasks.extend(model_tasks)

    dataset_predictions = HafniaDataset.from_samples_list(prediction_samples, info=prediction_dataset_info)
    return dataset_predictions
