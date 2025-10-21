import shutil
from pathlib import Path
from typing import List, Optional

import more_itertools
import polars as pl
from PIL import Image
from rich.progress import track

from hafnia.dataset.dataset_names import ColumnName, FieldName
from hafnia.dataset.hafnia_dataset import DatasetInfo, HafniaDataset, Sample, TaskInfo
from hafnia.dataset.primitives import Classification
from hafnia.utils import is_image_file


def import_image_classification_directory_tree(
    path_folder: Path,
    split: str,
    n_samples: Optional[int] = None,
) -> HafniaDataset:
    class_folder_paths = [path for path in path_folder.iterdir() if path.is_dir()]
    class_names = sorted([folder.name for folder in class_folder_paths])  # Sort for determinism

    # Gather all image paths per class
    path_images_per_class: List[List[Path]] = []
    for path_class_folder in class_folder_paths:
        per_class_images = []
        for path_image in list(path_class_folder.rglob("*.*")):
            if is_image_file(path_image):
                per_class_images.append(path_image)
        path_images_per_class.append(sorted(per_class_images))

    # Interleave to ensure classes are balanced in the output dataset for n_samples < total
    path_images = list(more_itertools.interleave_longest(*path_images_per_class))

    if n_samples is not None:
        path_images = path_images[:n_samples]

    samples = []
    for path_image_org in track(path_images, description="Convert 'image classification' dataset to Hafnia Dataset"):
        class_name = path_image_org.parent.name

        read_image = Image.open(path_image_org)
        width, height = read_image.size

        classifications = [Classification(class_name=class_name, class_idx=class_names.index(class_name))]
        sample = Sample(
            file_path=str(path_image_org.absolute()),
            width=width,
            height=height,
            split=split,
            classifications=classifications,
        )
        samples.append(sample)

    dataset_info = DatasetInfo(
        dataset_name="ImageClassificationFromDirectoryTree",
        tasks=[TaskInfo(primitive=Classification, class_names=class_names)],
    )

    hafnia_dataset = HafniaDataset.from_samples_list(samples_list=samples, info=dataset_info)
    return hafnia_dataset


def export_image_classification_directory_tree(
    dataset: HafniaDataset,
    path_output: Path,
    task_name: Optional[str] = None,
    clean_folder: bool = False,
) -> Path:
    task = dataset.info.get_task_by_task_name_and_primitive(task_name=task_name, primitive=Classification)

    samples = dataset.samples.with_columns(
        pl.col(task.primitive.column_name())
        .list.filter(pl.element().struct.field(FieldName.TASK_NAME) == task.name)
        .alias(task.primitive.column_name())
    )

    classification_counts = samples[task.primitive.column_name()].list.len()
    has_no_classification_samples = (classification_counts == 0).sum()
    if has_no_classification_samples > 0:
        raise ValueError(f"Some samples do not have a classification for task '{task.name}'.")

    has_multi_classification_samples = (classification_counts > 1).sum()
    if has_multi_classification_samples > 0:
        raise ValueError(f"Some samples have multiple classifications for task '{task.name}'.")

    if clean_folder:
        shutil.rmtree(path_output, ignore_errors=True)
    path_output.mkdir(parents=True, exist_ok=True)

    description = "Export Hafnia Dataset to directory tree"
    for sample_dict in track(samples.iter_rows(named=True), total=len(samples), description=description):
        classifications = sample_dict[task.primitive.column_name()]
        if len(classifications) != 1:
            raise ValueError("Each sample should have exactly one classification.")
        classification = classifications[0]
        class_name = classification[FieldName.CLASS_NAME].replace("/", "_")  # Avoid issues with subfolders
        path_class_folder = path_output / class_name
        path_class_folder.mkdir(parents=True, exist_ok=True)

        path_image_org = Path(sample_dict[ColumnName.FILE_PATH])
        path_image_new = path_class_folder / path_image_org.name
        shutil.copy2(path_image_org, path_image_new)

    return path_output
