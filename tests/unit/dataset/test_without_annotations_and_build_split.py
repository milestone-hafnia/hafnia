import json
from pathlib import Path
from typing import Sequence

import polars as pl
import pytest

from hafnia.dataset.dataset_names import SampleField, SplitName
from hafnia.dataset.dataset_recipe.dataset_recipe import DatasetRecipe
from hafnia.dataset.hafnia_dataset import HafniaDataset
from hafnia.dataset.hafnia_dataset_types import ClassInfo, DatasetInfo, Sample, TaskInfo
from hafnia.dataset.primitives import PRIMITIVE_COLUMN_NAMES
from hafnia.dataset.primitives.bbox import Bbox
from hafnia.dataset.primitives.classification import Classification

TAG_VALUE = "night"
ALL_SPLITS = (SplitName.TRAIN, SplitName.VAL, SplitName.TEST)


def make_dataset(path_folder: Path, name: str = "tiny", splits: Sequence[str] = ALL_SPLITS) -> HafniaDataset:
    """Two samples per split, each with a box, a classification, a tag and per-sample `meta`."""
    info = DatasetInfo(
        dataset_name=name,
        tasks=[
            TaskInfo(primitive=Bbox, name="object_detection", classes=[ClassInfo(name="Car"), ClassInfo(name="Truck")]),
            TaskInfo(primitive=Classification, name="Time of Day", classes=[ClassInfo(name="Day")]),
        ],
    )
    samples = [
        Sample(
            file_path=str(path_folder / "data" / f"{name}_{split}_{i}.jpg"),
            height=10,
            width=10,
            split=split,
            bboxes=[Bbox(top_left_x=0.1, top_left_y=0.1, width=0.2, height=0.2, class_name="Car", class_idx=0)],
            classifications=[Classification(class_name="Day", class_idx=0, task_name="Time of Day")],
            tags=[TAG_VALUE],
            meta={"camera": f"cam-{i}"},
        )
        for split in splits
        for i in range(2)
    ]
    return HafniaDataset.from_samples_list(samples, info=info)


def n_entries(samples: pl.DataFrame, column: str) -> int:
    return samples[column].list.len().fill_null(0).sum()


def recipe_from_written(dataset: HafniaDataset, path_folder: Path) -> DatasetRecipe:
    dataset.write_annotations(path_folder=path_folder)
    return DatasetRecipe.from_path(path_folder, check_for_images=False)


### without_annotations ###


def test_without_annotations_empties_every_label_and_tag(tmp_path: Path) -> None:
    dataset = make_dataset(tmp_path)
    for column in (SampleField.BBOXES, SampleField.CLASSIFICATIONS, SampleField.TAGS):
        assert n_entries(dataset.samples, column) > 0
    assert SampleField.META in dataset.samples.columns

    label_free = dataset.without_annotations()

    for column in [*PRIMITIVE_COLUMN_NAMES, SampleField.TAGS]:
        if column in label_free.samples.columns:
            assert n_entries(label_free.samples, column) == 0, column
    assert label_free.samples[SampleField.META].null_count() == len(label_free)
    assert label_free.samples.schema == dataset.samples.schema
    assert n_entries(dataset.samples, SampleField.BBOXES) > 0  # The input is not modified


def test_without_annotations_keeps_info_and_join_key(tmp_path: Path) -> None:
    dataset = make_dataset(tmp_path)
    label_free = dataset.without_annotations()

    assert label_free.info.tasks == dataset.info.tasks
    for column in (SampleField.SAMPLE_INDEX, SampleField.FILE_PATH):
        assert label_free.samples[column].to_list() == dataset.samples[column].to_list()


def test_without_annotations_writes_no_labels(tmp_path: Path) -> None:
    label_free = make_dataset(tmp_path).without_annotations()
    label_free.write_annotations(path_folder=tmp_path)

    rows = [json.loads(line) for line in (tmp_path / "annotations.jsonl").read_text().splitlines()]
    text = json.dumps(rows)
    assert len(rows) == len(label_free)
    assert "class_name" not in text
    assert TAG_VALUE not in text
    assert "cam-" not in text  # Per-sample meta values

    reloaded = HafniaDataset.from_path(tmp_path, check_for_images=False)
    assert reloaded.samples.schema[SampleField.META] == label_free.samples.schema[SampleField.META]


def test_without_annotations_skips_absent_columns(tmp_path: Path) -> None:
    dataset = make_dataset(tmp_path)
    dataset = dataset.update_samples(dataset.samples.drop(SampleField.CLASSIFICATIONS, SampleField.META))

    label_free = dataset.without_annotations()

    assert SampleField.CLASSIFICATIONS not in label_free.samples.columns
    assert n_entries(label_free.samples, SampleField.BBOXES) == 0


### DatasetRecipe.build(split=...) ###


def test_build_split_keeps_only_that_split(tmp_path: Path) -> None:
    recipe = recipe_from_written(make_dataset(tmp_path), tmp_path)
    full = recipe.build(download_files=False)

    test_only = recipe.build(download_files=False, split=SplitName.TEST)

    assert set(test_only.samples[SampleField.SPLIT].to_list()) == {SplitName.TEST}
    expected = full.samples.filter(pl.col(SampleField.SPLIT) == SplitName.TEST)[SampleField.SAMPLE_INDEX]
    assert test_only.samples[SampleField.SAMPLE_INDEX].to_list() == expected.to_list()


def test_build_without_split_keeps_everything(tmp_path: Path) -> None:
    dataset = make_dataset(tmp_path)
    recipe = recipe_from_written(dataset, tmp_path)
    assert len(recipe.build(download_files=False)) == len(dataset)


def test_build_empty_split_raises(tmp_path: Path) -> None:
    dataset = make_dataset(tmp_path, splits=(SplitName.TRAIN, SplitName.VAL))
    recipe = recipe_from_written(dataset, tmp_path)

    with pytest.raises(ValueError, match=rf"'{SplitName.TEST}'.*{len(dataset)} samples"):
        recipe.build(download_files=False, split=SplitName.TEST)


def test_build_split_on_merged_recipe(tmp_path: Path) -> None:
    recipes = [
        recipe_from_written(make_dataset(tmp_path / name, name=name), tmp_path / name) for name in ("tiny-a", "tiny-b")
    ]
    merged = DatasetRecipe.from_merger(recipes).build(download_files=False, split=SplitName.TEST)

    assert set(merged.samples[SampleField.SPLIT].to_list()) == {SplitName.TEST}
    assert merged.samples.group_by(SampleField.DATASET_NAME).len().sort(SampleField.DATASET_NAME).rows() == [
        ("tiny-a", 2),
        ("tiny-b", 2),
    ]


def test_build_split_filters_after_operations(tmp_path: Path) -> None:
    recipe = recipe_from_written(make_dataset(tmp_path), tmp_path).splits_by_ratios(
        {SplitName.TRAIN: 0.5, SplitName.TEST: 0.5}, seed=0
    )
    expected = recipe.build(download_files=False).create_split_dataset(SplitName.TEST)

    test_only = recipe.build(download_files=False, split=SplitName.TEST)

    assert test_only.samples[SampleField.FILE_PATH].to_list() == expected.samples[SampleField.FILE_PATH].to_list()
