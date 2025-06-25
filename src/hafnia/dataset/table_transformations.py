from pathlib import Path
from typing import List, Optional, Type

import polars as pl
from tqdm import tqdm

from hafnia.dataset import table_transformations
from hafnia.dataset.dataset_names import (
    FILENAME_ANNOTATIONS_JSONL,
    FILENAME_ANNOTATIONS_PARQUET,
    FieldName,
)
from hafnia.dataset.primitives import PRIMITIVE_TYPES
from hafnia.dataset.primitives.classification import Classification
from hafnia.dataset.primitives.primitive import Primitive
from hafnia.log import user_logger


def create_primitive_table(
    samples_table: pl.DataFrame, PrimitiveType: Type[Primitive], keep_sample_data: bool = False
) -> Optional[pl.DataFrame]:
    """
    Returns a DataFrame with objects of the specified primitive type.
    """
    column_name = PrimitiveType.column_name()
    has_primitive_column = (column_name in samples_table.columns) and (
        samples_table[column_name].dtype == pl.List(pl.Struct)
    )
    if not has_primitive_column:
        return None

    # Remove frames without objects
    remove_no_object_frames = samples_table.filter(pl.col(column_name).list.len() > 0)

    if keep_sample_data:
        # Drop other primitive columns to avoid conflicts
        drop_columns = set(PRIMITIVE_TYPES) - {PrimitiveType, Classification}
        remove_no_object_frames = remove_no_object_frames.drop(*[primitive.column_name() for primitive in drop_columns])
        # Rename columns "height", "width" and "meta" for sample to avoid conflicts with object fields names
        remove_no_object_frames = remove_no_object_frames.rename(
            {"height": "image.height", "width": "image.width", "meta": "image.meta"}
        )
        objects_df = remove_no_object_frames.explode(column_name).unnest(column_name)
    else:
        objects_df = remove_no_object_frames.select(pl.col(column_name).explode().struct.unnest())
    return objects_df


def filter_table_for_class_names(
    samples_table: pl.DataFrame, class_names: List[str], PrimitiveType: Type[Primitive]
) -> Optional[pl.DataFrame]:
    table_with_selected_class_names = samples_table.filter(
        pl.col(PrimitiveType.column_name())
        .list.eval(pl.element().struct.field(FieldName.CLASS_NAME).is_in(class_names))
        .list.any()
    )

    return table_with_selected_class_names


def split_primitive_columns_by_task_name(
    samples_table: pl.DataFrame,
    coordinate_types: Optional[List[Type[Primitive]]] = None,
) -> pl.DataFrame:
    """
    Convert Primitive columns such as "objects" (Bbox) into a column for each task name.
    For example, if the "objects" column (containing Bbox objects) has tasks "task1" and "task2".


    This:
    ─┬────────────┬─
     ┆ objects    ┆
     ┆ ---        ┆
     ┆ list[struc ┆
     ┆ t[11]]     ┆
    ═╪════════════╪═
    becomes this:
    ─┬────────────┬────────────┬─
     ┆ objects.   ┆ objects.   ┆
     ┆ task1      ┆ task2      ┆
     ┆ ---        ┆ ---        ┆
     ┆ list[struc ┆ list[struc ┆
     ┆ t[11]]     ┆ t[13]]     ┆
    ═╪════════════╪════════════╪═

    """
    coordinate_types = coordinate_types or PRIMITIVE_TYPES
    for PrimitiveType in coordinate_types:
        col_name = PrimitiveType.column_name()

        if col_name not in samples_table.columns:
            continue

        if samples_table[col_name].dtype != pl.List(pl.Struct):
            continue

        task_names = samples_table[col_name].explode().struct.field(FieldName.TASK_NAME).unique().to_list()
        samples_table = samples_table.with_columns(
            [
                pl.col(col_name)
                .list.filter(pl.element().struct.field(FieldName.TASK_NAME).eq(task_name))
                .alias(f"{col_name}.{task_name}")
                for task_name in task_names
            ]
        )
        samples_table = samples_table.drop(col_name)
    return samples_table


def read_table_from_path(path: Path) -> pl.DataFrame:
    path_annotations = path / FILENAME_ANNOTATIONS_PARQUET
    if path_annotations.exists():
        user_logger.info(f"Reading dataset annotations from Parquet file: {path_annotations}")
        return pl.read_parquet(path_annotations)

    path_annotations_jsonl = path / FILENAME_ANNOTATIONS_JSONL
    if path_annotations_jsonl.exists():
        user_logger.info(f"Reading dataset annotations from JSONL file: {path_annotations_jsonl}")
        return pl.read_ndjson(path_annotations_jsonl)

    raise FileNotFoundError(
        f"Unable to read annotations. No json file '{path_annotations.name}' or Parquet file '{{path_annotations.name}} in in '{path}'."
    )


def check_image_paths(table: pl.DataFrame) -> bool:
    missing_files = []
    for org_path in tqdm(table["file_name"].to_list(), desc="Check image paths"):
        org_path = Path(org_path)
        if not org_path.exists():
            missing_files.append(org_path)

    if len(missing_files) > 0:
        user_logger.warning(f"Missing files: {len(missing_files)}. Show first 5:")
        for missing_file in missing_files[:5]:
            user_logger.warning(f" - {missing_file}")
        raise FileNotFoundError(f"Some files are missing in the dataset: {len(missing_files)} files not found.")

    return True


def unnest_classification_tasks(table: pl.DataFrame, strict: bool = True) -> pl.DataFrame:
    """
    Unnest classification tasks in table.
    Classificiations tasks are all stored in the same column in the HafniaDataset table.
    This function splits them into separate columns for each task name.

    Type is converted from a list of structs (pl.List[pl.Struct]) to a struct (pl.Struct) column.

    Converts classification column from this:
       ─┬─────────────────┬─
        ┆ classifications ┆
        ┆ ---             ┆
        ┆ list[struct[6]] ┆
       ═╪═════════════════╪═

    For example, if the classification column has tasks "task1" and "task2",
       ─┬──────────────────┬──────────────────┬─
        ┆ classifications. ┆ classifications. ┆
        ┆ task1            ┆ task2            ┆
        ┆ ---              ┆ ---              ┆
        ┆ struct[6]        ┆ struct[6]        ┆
       ═╪══════════════════╪══════════════════╪═

    """
    coordinate_types = [Classification]
    table_out = table_transformations.split_primitive_columns_by_task_name(table, coordinate_types=coordinate_types)

    classification_columns = [c for c in table_out.columns if c.startswith(Classification.column_name() + ".")]
    for classification_column in classification_columns:
        has_multiple_items_per_sample = all(table_out[classification_column].list.len() > 1)
        if has_multiple_items_per_sample:
            if strict:
                raise ValueError(
                    f"Column {classification_column} has multiple items per sample, but expected only one item."
                )
            else:
                user_logger.warning(
                    f"Warning: Unnesting of column '{classification_column}' is skipped because it has multiple items per sample."
                )

    table_out = table_out.with_columns([pl.col(c).list.first() for c in classification_columns])
    return table_out
