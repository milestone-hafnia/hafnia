from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Optional, Type

import polars as pl
import rich
from rich import print as rprint
from rich.progress import track
from rich.table import Table

from hafnia.dataset.dataset_names import ColumnName, FieldName, SplitName
from hafnia.dataset.operations.table_transformations import create_primitive_table
from hafnia.dataset.primitives import PRIMITIVE_TYPES
from hafnia.log import user_logger

if TYPE_CHECKING:  # Using 'TYPE_CHECKING' to avoid circular imports during type checking
    from hafnia.dataset.hafnia_dataset import HafniaDataset
    from hafnia.dataset.primitives.primitive import Primitive


def split_counts(dataset: HafniaDataset) -> Dict[str, int]:
    """
    Returns a dictionary with the counts of samples in each split of the dataset.
    """
    return dict(dataset.samples[ColumnName.SPLIT].value_counts().iter_rows())


def class_counts_for_task(
    dataset: HafniaDataset,
    primitive: Optional[Type[Primitive]] = None,
    task_name: Optional[str] = None,
) -> Dict[str, int]:
    """
    Determines class name counts for a specific task in the dataset.

    The counts are returned as a dictionary where keys are class names and values are their respective counts.
    Note that class names with zero counts are included in the dictionary and that
    the order of the dictionary matches the class idx.

    >>> counts = dataset.class_counts_for_task(primitive=Bbox)
    >>> print(counts)
    {
        'person': 0,  # Note: Zero count classes are included to maintain order with class idx
        'car': 500,
        'bicycle': 0,
        'bus': 100,
        'truck': 150,
        'motorcycle': 50,
    }
    """
    task = dataset.info.get_task_by_task_name_and_primitive(task_name=task_name, primitive=primitive)
    class_counts_df = (
        dataset.samples[task.primitive.column_name()]
        .explode()
        .struct.unnest()
        .filter(pl.col(FieldName.TASK_NAME) == task.name)[FieldName.CLASS_NAME]
        .value_counts()
    )

    # Initialize counts with zero for all classes to ensure zero-count classes are included
    # and to have class names in the order of class idx
    class_counts = {name: 0 for name in task.class_names or []}
    class_counts.update(dict(class_counts_df.iter_rows()))

    return class_counts


def class_counts_all(dataset: HafniaDataset) -> Dict[str, int]:
    """
    Get class counts for all tasks in the dataset.
    The counts are returned as a dictionary where keys are in the format
    '{primitive_name}/{task_name}/{class_name}' and values are their respective counts.

    Example:
    >>> counts = dataset.class_counts_all()
    >>> print(counts)
    {
        objects/bboxes/car: 500
        objects/bboxes/person: 0
        classifications/weather/sunny: 300
        classifications/weather/rainy: 0
        ...
    }
    """
    class_counts = {}
    for task in dataset.info.tasks:
        if task.class_names is None:
            raise ValueError(f"Task '{task.name}' does not have class names defined.")
        class_counts_task = dataset.class_counts_for_task(primitive=task.primitive, task_name=task.name)

        for class_idx, (class_name, count) in enumerate(class_counts_task.items()):
            count_name = f"{task.primitive.__name__}/{task.name}/{class_name}"
            class_counts[count_name] = count

    return class_counts


def print_stats(dataset: HafniaDataset) -> None:
    """
    Prints verbose statistics about the dataset, including dataset name, version,
    number of samples, and detailed counts of samples and tasks.
    """
    table_base = Table(title="Dataset Statistics", show_lines=True, box=rich.box.SIMPLE)
    table_base.add_column("Property", style="cyan")
    table_base.add_column("Value")
    table_base.add_row("Dataset Name", dataset.info.dataset_name)
    table_base.add_row("Version", dataset.info.version)
    table_base.add_row("Number of samples", str(len(dataset.samples)))
    rprint(table_base)
    print_sample_and_task_counts(dataset)
    print_class_distribution(dataset)


def print_class_distribution(dataset: HafniaDataset) -> None:
    """
    Prints the class distribution for each task in the dataset.
    """
    for task in dataset.info.tasks:
        if task.class_names is None:
            raise ValueError(f"Task '{task.name}' does not have class names defined.")
        class_counts = dataset.class_counts_for_task(primitive=task.primitive, task_name=task.name)

        # Print class distribution
        rich_table = Table(title=f"Class Count for '{task.primitive.__name__}/{task.name}'", show_lines=False)
        rich_table.add_column("Class Name", style="cyan")
        rich_table.add_column("Class Idx", style="cyan")
        rich_table.add_column("Count", justify="right")
        for class_name, count in class_counts.items():
            class_idx = task.class_names.index(class_name)  # Get class idx from task info
            rich_table.add_row(class_name, str(class_idx), str(count))
        rprint(rich_table)


def print_sample_and_task_counts(dataset: HafniaDataset) -> None:
    """
    Prints a table with sample counts and task counts for each primitive type
    in total and for each split (train, val, test).
    """
    from hafnia.dataset.operations.table_transformations import create_primitive_table
    from hafnia.dataset.primitives import PRIMITIVE_TYPES

    splits_sets = {
        "All": SplitName.valid_splits(),
        "Train": [SplitName.TRAIN],
        "Validation": [SplitName.VAL],
        "Test": [SplitName.TEST],
    }
    rows = []
    for split_name, splits in splits_sets.items():
        dataset_split = dataset.create_split_dataset(splits)
        table = dataset_split.samples
        row = {}
        row["Split"] = split_name
        row["Sample "] = str(len(table))
        for PrimitiveType in PRIMITIVE_TYPES:
            column_name = PrimitiveType.column_name()
            objects_df = create_primitive_table(table, PrimitiveType=PrimitiveType, keep_sample_data=False)
            if objects_df is None:
                continue
            for (task_name,), object_group in objects_df.group_by(FieldName.TASK_NAME):
                count = len(object_group[FieldName.CLASS_NAME])
                row[f"{PrimitiveType.__name__}\n{task_name}"] = str(count)
        rows.append(row)

    rich_table = Table(title="Dataset Statistics", show_lines=True, box=rich.box.SIMPLE)
    for i_row, row in enumerate(rows):
        if i_row == 0:
            for column_name in row.keys():
                rich_table.add_column(column_name, justify="left", style="cyan")
        rich_table.add_row(*[str(value) for value in row.values()])
    rprint(rich_table)


def check_dataset(dataset: HafniaDataset):
    """
    Performs various checks on the dataset to ensure its integrity and consistency.
    Raises errors if any issues are found.
    """
    from hafnia.dataset.hafnia_dataset import Sample

    user_logger.info("Checking Hafnia dataset...")
    assert isinstance(dataset.info.dataset_name, str) and len(dataset.info.dataset_name) > 0

    sample_dataset = dataset.create_sample_dataset()
    if len(sample_dataset) == 0:
        raise ValueError("The dataset does not include a sample dataset")

    actual_splits = dataset.samples.select(pl.col(ColumnName.SPLIT)).unique().to_series().to_list()
    expected_splits = SplitName.valid_splits()
    if set(actual_splits) != set(expected_splits):
        raise ValueError(f"Expected all splits '{expected_splits}' in dataset, but got '{actual_splits}'. ")

    dataset.check_dataset_tasks()

    expected_tasks = dataset.info.tasks
    distribution = dataset.info.distributions or []
    distribution_names = [task.name for task in distribution]
    # Check that tasks found in the 'dataset.table' matches the tasks defined in 'dataset.info.tasks'
    for PrimitiveType in PRIMITIVE_TYPES:
        column_name = PrimitiveType.column_name()
        if column_name not in dataset.samples.columns:
            continue
        objects_df = create_primitive_table(dataset.samples, PrimitiveType=PrimitiveType, keep_sample_data=False)
        if objects_df is None:
            continue
        for (task_name,), object_group in objects_df.group_by(FieldName.TASK_NAME):
            has_task = any([t for t in expected_tasks if t.name == task_name and t.primitive == PrimitiveType])
            if has_task or (task_name in distribution_names):
                continue
            class_names = object_group[FieldName.CLASS_NAME].unique().to_list()
            raise ValueError(
                f"Task name '{task_name}' for the '{PrimitiveType.__name__}' primitive is missing in "
                f"'dataset.info.tasks' for dataset '{task_name}'. Missing task has the following "
                f"classes: {class_names}. "
            )

    for sample_dict in track(dataset, description="Checking samples in dataset"):
        sample = Sample(**sample_dict)  # noqa: F841


def check_dataset_tasks(dataset: HafniaDataset):
    """
    Checks that the tasks defined in 'dataset.info.tasks' are consistent with the data in 'dataset.samples'.
    """
    dataset.info.check_for_duplicate_task_names()

    for task in dataset.info.tasks:
        primitive = task.primitive.__name__
        column_name = task.primitive.column_name()
        primitive_column = dataset.samples[column_name]
        msg_something_wrong = (
            f"Something is wrong with the defined tasks ('info.tasks') in dataset '{dataset.info.dataset_name}'. \n"
            f"For '{primitive=}' and '{task.name=}' "
        )
        if primitive_column.dtype == pl.Null:
            raise ValueError(msg_something_wrong + "the column is 'Null'. Please check the dataset.")

        if len(dataset) > 0:  # Check only performed for non-empty datasets
            primitive_table = (
                primitive_column.explode().struct.unnest().filter(pl.col(FieldName.TASK_NAME) == task.name)
            )
            if primitive_table.is_empty():
                raise ValueError(
                    msg_something_wrong
                    + f"the column '{column_name}' has no {task.name=} objects. Please check the dataset."
                )

            actual_classes = set(primitive_table[FieldName.CLASS_NAME].unique().to_list())
            if task.class_names is None:
                raise ValueError(
                    msg_something_wrong
                    + f"the column '{column_name}' with {task.name=} has no defined classes. Please check the dataset."
                )
            defined_classes = set(task.class_names)

            if not actual_classes.issubset(defined_classes):
                raise ValueError(
                    msg_something_wrong
                    + f"the column '{column_name}' with {task.name=} we expected the actual classes in the dataset to \n"
                    f"to be a subset of the defined classes\n\t{actual_classes=} \n\t{defined_classes=}."
                )
            # Check class_indices
            mapped_indices = primitive_table[FieldName.CLASS_NAME].map_elements(
                lambda x: task.class_names.index(x), return_dtype=pl.Int64
            )
            table_indices = primitive_table[FieldName.CLASS_IDX]

            error_msg = msg_something_wrong + (
                f"class indices in '{FieldName.CLASS_IDX}' column does not match classes ordering in 'task.class_names'"
            )
            assert mapped_indices.equals(table_indices), error_msg
