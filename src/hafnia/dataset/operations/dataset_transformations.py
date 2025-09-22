"""
Hafnia dataset transformations that takes and returns a HafniaDataset object.

All functions here will have a corresponding function in both the HafniaDataset class
and a corresponding RecipeTransform class in the `data_recipe/recipe_transformations.py` file.

This allows each function to be used in three ways:

```python
from hafnia.dataset.operations import dataset_transformations
from hafnia.dataset.hafnia_dataset import HafniaDataset
from hafnia.dataset.data_recipe.recipe_transformations import SplitByRatios

splits_by_ratios = {"train": 0.8, "val": 0.1, "test": 0.1}

# Option 1: Using the function directly
dataset = recipe_transformations.splits_by_ratios(dataset, split_ratios=splits_by_ratios)

# Option 2: Using the method of the HafniaDataset class
dataset = dataset.splits_by_ratios(split_ratios=splits_by_ratios)

# Option 3: Using the RecipeTransform class
serializable_transform = SplitByRatios(split_ratios=splits_by_ratios)
dataset = serializable_transform(dataset)
```

Tests will ensure that all functions in this file will have a corresponding function in the
HafniaDataset class and a RecipeTransform class in the `data_recipe/recipe_transformations.py` file and
that the signatures match.
"""

import json
import textwrap
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Set, Tuple, Type, Union

import cv2
import more_itertools
import numpy as np
import polars as pl
from PIL import Image
from tqdm import tqdm

from hafnia.dataset import dataset_helpers
from hafnia.dataset.dataset_names import TAG_REMOVE_CLASS, FieldName
from hafnia.dataset.primitives import get_primitive_type_from_string
from hafnia.dataset.primitives.primitive import Primitive
from hafnia.utils import remove_duplicates_preserve_order

if TYPE_CHECKING:  # Using 'TYPE_CHECKING' to avoid circular imports during type checking
    from hafnia.dataset.hafnia_dataset import HafniaDataset, TaskInfo


### Image transformations ###
class AnonymizeByPixelation:
    def __init__(self, resize_factor: float = 0.10):
        self.resize_factor = resize_factor

    def __call__(self, frame: np.ndarray) -> np.ndarray:
        org_size = frame.shape[:2]
        frame = cv2.resize(frame, (0, 0), fx=self.resize_factor, fy=self.resize_factor)
        frame = cv2.resize(frame, org_size[::-1], interpolation=cv2.INTER_NEAREST)
        return frame


def transform_images(
    dataset: "HafniaDataset",
    transform: Callable[[np.ndarray], np.ndarray],
    path_output: Path,
) -> "HafniaDataset":
    new_paths = []
    path_image_folder = path_output / "data"
    path_image_folder.mkdir(parents=True, exist_ok=True)

    for org_path in tqdm(dataset.samples["file_name"].to_list(), desc="Transform images"):
        org_path = Path(org_path)
        if not org_path.exists():
            raise FileNotFoundError(f"File {org_path} does not exist in the dataset.")

        image = np.array(Image.open(org_path))
        image_transformed = transform(image)
        new_path = dataset_helpers.save_image_with_hash_name(image_transformed, path_image_folder)

        if not new_path.exists():
            raise FileNotFoundError(f"Transformed file {new_path} does not exist in the dataset.")
        new_paths.append(str(new_path))

    table = dataset.samples.with_columns(pl.Series(new_paths).alias("file_name"))
    return dataset.update_samples(table)


def get_task_info_from_task_name_and_primitive(
    tasks: List["TaskInfo"],
    task_name: Optional[str] = None,
    primitive: Union[None, str, Type[Primitive]] = None,
) -> "TaskInfo":
    if len(tasks) == 0:
        raise ValueError("Dataset has no tasks defined.")

    tasks_str = "\n".join([f"\t{task.__repr__()}" for task in tasks])
    if task_name is None and primitive is None:
        if len(tasks) == 1:
            return tasks[0]
        else:
            raise ValueError(
                "For multiple tasks, you will need to specify 'task_name' or 'type_primitive' "
                "to return a unique task. The dataset contains the following tasks: \n" + tasks_str
            )

    if isinstance(primitive, str):
        primitive = get_primitive_type_from_string(primitive)

    tasks_filtered = tasks
    if primitive is None:
        tasks_filtered = [task for task in tasks if task.name == task_name]

        if len(tasks_filtered) == 0:
            raise ValueError(f"No task found with {task_name=}. Available tasks: \n {tasks_str}")

        unique_primitives = set(task.primitive for task in tasks_filtered)
        if len(unique_primitives) > 1:
            raise ValueError(
                f"Found multiple tasks with {task_name=} using different primitives {unique_primitives=}. "
                "Please specify the primitive type to make it unique. "
                f"The dataset contains the following tasks: \n {tasks_str}"
            )
        primitive = list(unique_primitives)[0]

    if task_name is None:
        tasks_filtered = [task for task in tasks if task.primitive == primitive]
        if len(tasks_filtered) == 0:
            raise ValueError(f"No task found with {primitive=}. Available tasks: \n {tasks_str}")

        unique_task_names = set(task.name for task in tasks_filtered)
        if len(unique_task_names) > 1:
            raise ValueError(
                f"Found multiple tasks with {primitive=} using different task names {unique_task_names=}. "
                "Please specify the 'task_name' to make it unique."
                f"The dataset contains the following tasks: \n {tasks_str}"
            )
        task_name = list(unique_task_names)[0]

    tasks_filtered = [task for task in tasks_filtered if task.primitive == primitive and task.name == task_name]
    if len(tasks_filtered) == 0:
        raise ValueError(f"No task found with {task_name=} and {primitive=}. Available tasks: \n {tasks_str}")

    if len(tasks_filtered) > 1:
        raise ValueError(
            f"Multiple tasks found with {task_name=} and {primitive=}. "
            f"This should never happen. The dataset contains the following tasks: \n {tasks_str}"
        )
    task = tasks_filtered[0]
    return task


def class_mapper_strict(
    dataset: "HafniaDataset",
    strict_class_mapping: Dict[str, str],
    primitive: Optional[Type[Primitive]] = None,
    task_name: Optional[str] = None,
) -> "HafniaDataset":
    from hafnia.dataset.hafnia_dataset import HafniaDataset

    task = dataset.info.get_task_by_task_name_and_primitive(task_name=task_name, primitive=primitive)

    current_class_names = task.class_names or []
    class_names_mapped = list(strict_class_mapping.keys())
    missing_class_names = set(current_class_names) - set(class_names_mapped)
    if len(missing_class_names) > 0:
        error_msg = f"""\
        The specified class mapping is not a strict mapping - meaning that all class names have not 
        been mapped to a new class name.
        In the current mapping, the following classes {list(missing_class_names)} have not been mapped.
        The currently specified mapping is:
        {json.dumps(strict_class_mapping, indent=2)}
        A strict mapping will replace all old class names (dictionary keys) to new class names (dictionary values).
        Please update the mapping to include all class names from the dataset task '{task.name}'.
        To keep class map to the same name e.g. 'person' = 'person' 
        or remove class by using the '__REMOVE__' key, e.g. 'person': '__REMOVE__'."""
        raise ValueError(textwrap.dedent(error_msg))

    new_class_names = remove_duplicates_preserve_order(strict_class_mapping.values())

    if TAG_REMOVE_CLASS in new_class_names:
        # Move __REMOVE__ to the end of the list if it exists
        new_class_names.append(new_class_names.pop(new_class_names.index(TAG_REMOVE_CLASS)))
    name_2_idx_mapping: Dict[str, int] = {name: idx for idx, name in enumerate(new_class_names)}

    samples = dataset.samples
    samples_updated = samples.with_columns(
        pl.col(task.primitive.column_name())
        .list.eval(
            pl.element().struct.with_fields(
                pl.when(pl.field(FieldName.TASK_NAME) == task.name)
                .then(pl.field(FieldName.CLASS_NAME).replace_strict(strict_class_mapping))
                .otherwise(pl.field(FieldName.CLASS_NAME))
                .alias(FieldName.CLASS_NAME)
            )
        )
        .alias(task.primitive.column_name())
    )

    # Update class indices too
    samples_updated = samples_updated.with_columns(
        pl.col(task.primitive.column_name())
        .list.eval(
            pl.element().struct.with_fields(
                pl.when(pl.field(FieldName.TASK_NAME) == task.name)
                .then(pl.field(FieldName.CLASS_NAME).replace_strict(name_2_idx_mapping))
                .otherwise(pl.field(FieldName.CLASS_IDX))
                .alias(FieldName.CLASS_IDX)
            )
        )
        .alias(task.primitive.column_name())
    )

    if TAG_REMOVE_CLASS in new_class_names:  # Remove class_names that are mapped to REMOVE_CLASS
        samples_updated = samples_updated.with_columns(
            pl.col(task.primitive.column_name())
            .list.filter(pl.element().struct.field(FieldName.CLASS_NAME) != TAG_REMOVE_CLASS)
            .alias(task.primitive.column_name())
        )
        new_class_names = [c for c in new_class_names if c != TAG_REMOVE_CLASS]

    new_task = task.model_copy(deep=True)
    new_task.class_names = new_class_names
    dataset_info = dataset.info.replace_task(old_task=task, new_task=new_task)
    return HafniaDataset(info=dataset_info, samples=samples_updated)


def rename_task(
    dataset: "HafniaDataset",
    old_task_name: str,
    new_task_name: str,
) -> "HafniaDataset":
    from hafnia.dataset.hafnia_dataset import HafniaDataset

    old_task = dataset.info.get_task_by_name(task_name=old_task_name)
    new_task = old_task.model_copy(deep=True)
    new_task.name = new_task_name
    samples = dataset.samples.with_columns(
        pl.col(old_task.primitive.column_name())
        .list.eval(
            pl.element().struct.with_fields(
                pl.field(FieldName.TASK_NAME).replace(old_task.name, new_task.name).alias(FieldName.TASK_NAME)
            )
        )
        .alias(new_task.primitive.column_name())
    )

    dataset_info = dataset.info.replace_task(old_task=old_task, new_task=new_task)
    return HafniaDataset(info=dataset_info, samples=samples)


def select_samples_by_class_name(
    dataset: "HafniaDataset",
    name: Union[List[str], str],
    task_name: Optional[str] = None,
    primitive: Optional[Type[Primitive]] = None,
) -> "HafniaDataset":
    task, class_names = _validate_inputs_select_samples_by_class_name(
        dataset=dataset,
        name=name,
        task_name=task_name,
        primitive=primitive,
    )

    samples = dataset.samples.filter(
        pl.col(task.primitive.column_name())
        .list.eval(
            pl.element().struct.field(FieldName.CLASS_NAME).is_in(class_names)
            & (pl.element().struct.field(FieldName.TASK_NAME) == task.name)
        )
        .list.any()
    )

    dataset_updated = dataset.update_samples(samples)
    return dataset_updated


def _validate_inputs_select_samples_by_class_name(
    dataset: "HafniaDataset",
    name: Union[List[str], str],
    task_name: Optional[str] = None,
    primitive: Optional[Type[Primitive]] = None,
) -> Tuple["TaskInfo", Set[str]]:
    if isinstance(name, str):
        name = [name]
    names = set(name)

    # Check that specified names are available in at least one of the tasks
    available_names_across_tasks = set(more_itertools.flatten([t.class_names for t in dataset.info.tasks]))
    missing_class_names_across_tasks = names - available_names_across_tasks
    if len(missing_class_names_across_tasks) > 0:
        raise ValueError(
            f"The specified names {list(names)} have not been found in any of the tasks. "
            f"Available class names: {available_names_across_tasks}"
        )

    # Auto infer task if task_name and primitive are not provided
    if task_name is None and primitive is None:
        tasks_with_names = [t for t in dataset.info.tasks if names.issubset(t.class_names or [])]
        if len(tasks_with_names) == 0:
            raise ValueError(
                f"The specified names {list(names)} have not been found in any of the tasks. "
                f"Available class names: {available_names_across_tasks}"
            )
        if len(tasks_with_names) > 1:
            raise ValueError(
                f"Found multiple tasks containing the specified names {list(names)}. "
                f"Specify either 'task_name' or 'primitive' to only select from one task. "
                f"Tasks containing all provided names: {[t.name for t in tasks_with_names]}"
            )

        task = tasks_with_names[0]

    else:
        task = get_task_info_from_task_name_and_primitive(
            tasks=dataset.info.tasks,
            task_name=task_name,
            primitive=primitive,
        )

    task_class_names = set(task.class_names or [])
    missing_class_names = names - task_class_names
    if len(missing_class_names) > 0:
        raise ValueError(
            f"The specified names {list(missing_class_names)} have not been found for the '{task.name}' task. "
            f"Available class names: {task_class_names}"
        )

    return task, names
