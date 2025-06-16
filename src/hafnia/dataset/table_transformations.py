from pathlib import Path
from typing import Optional, Type

import polars as pl
from tqdm import tqdm

from hafnia.dataset.base_types import Primitive
from hafnia.dataset.dataset_names import FILENAME_ANNOTATIONS_JSONL, FILENAME_ANNOTATIONS_PARQUET
from hafnia.dataset.shape_primitives import COORDINATE_TYPES, Classification


def objects_dataframe(
    table: pl.DataFrame, PrimitiveType: Type[Primitive], keep_sample_data: bool = False
) -> Optional[pl.DataFrame]:
    """
    Returns a DataFrame with objects of the specified primitive type.
    """
    column_name = PrimitiveType.column_name()
    has_primitive_column = (column_name in table.columns) and (table[column_name].dtype == pl.List(pl.Struct))
    if not has_primitive_column:
        return None

    # Remove frames without objects
    remove_no_object_frames = table.filter(pl.col(column_name).list.len() > 0)

    if keep_sample_data:
        # Drop other primitive columns to avoid conflicts
        drop_columns = set(COORDINATE_TYPES) - {PrimitiveType, Classification}
        remove_no_object_frames = remove_no_object_frames.drop(*[primitive.column_name() for primitive in drop_columns])
        # Rename columns "height", "width" and "meta" for sample to avoid conflicts with object fields names
        remove_no_object_frames = remove_no_object_frames.rename(
            {"height": "image.height", "width": "image.width", "meta": "image.meta"}
        )
        remove_no_object_frames = remove_no_object_frames.explode(column_name).unnest(column_name)
    else:
        objects_df = remove_no_object_frames.select(pl.col(column_name).explode().struct.unnest())
    return objects_df


def read_table_from_path(path: Path) -> pl.DataFrame:
    path_annotations = path / FILENAME_ANNOTATIONS_PARQUET
    if path_annotations.exists():
        print(f"Reading dataset annotations from Parquet file: {path_annotations}")
        return pl.read_parquet(path_annotations)

    path_annotations_jsonl = path / FILENAME_ANNOTATIONS_JSONL
    if path_annotations_jsonl.exists():
        print(f"Reading dataset annotations from JSONL file: {path_annotations_jsonl}")
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
        print(f"Missing files: {len(missing_files)}. Show first 5:")
        for missing_file in missing_files[:5]:
            print(f" - {missing_file}")
        raise FileNotFoundError(f"Some files are missing in the dataset: {len(missing_files)} files not found.")

    return True
