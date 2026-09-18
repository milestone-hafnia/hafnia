from datetime import datetime
from typing import List, Optional, Type

import polars as pl
import pytest

from hafnia.dataset.dataset_names import PrimitiveField, SampleField
from hafnia.dataset.hafnia_dataset import HafniaDataset
from hafnia.dataset.hafnia_dataset_types import ClassInfo, DatasetInfo, Sample, TaskInfo
from hafnia.dataset.operations import table_transformations
from hafnia.dataset.operations.table_transformations import unnest_classification_tasks
from hafnia.dataset.primitives import (
    Bbox,
    Bitmask,
    Classification,
    KeyPoint,
    Point,
    Skeleton,
    SkeletonEdge,
    SkeletonTemplate,
)
from hafnia.dataset.primitives.primitive import Primitive
from tests import helper_testing


@pytest.mark.parametrize("dataset_name", helper_testing.MICRO_DATASETS)
def test_create_primitive_table(dataset_name: str):
    hafnia_dataset = helper_testing.get_micro_hafnia_dataset(dataset_name=dataset_name, force_update=False)
    hafnia_dataset.samples

    PrimitiveTypes = [Classification, Bbox, Bitmask]

    for PrimitiveType in PrimitiveTypes:
        n_primitive_fields = len(PrimitiveType.model_fields)
        only_primitives = table_transformations.create_primitive_table(
            samples_table=hafnia_dataset.samples,
            PrimitiveType=PrimitiveType,  # type: ignore[type-abstract]
            keep_sample_data=False,
        )
        if only_primitives is not None:
            assert len(only_primitives.columns) <= n_primitive_fields

        all_columns = table_transformations.create_primitive_table(
            samples_table=hafnia_dataset.samples,
            PrimitiveType=PrimitiveType,  # type: ignore[type-abstract]
            keep_sample_data=True,
        )

        if all_columns is not None:
            assert len(all_columns.columns) > n_primitive_fields


def test_filter_table_for_class_names():
    hafnia_dataset = helper_testing.get_micro_hafnia_dataset(dataset_name="micro-tiny-dataset", force_update=False)

    n_samples_before_filtering = len(hafnia_dataset.samples)
    table_after = table_transformations.filter_table_for_class_names(
        samples_table=hafnia_dataset.samples,
        class_names=["Vehicle.Car"],
        PrimitiveType=Bbox,
    )

    n_samples_after_filtering = len(table_after)
    assert n_samples_after_filtering < n_samples_before_filtering


def test_split_primitive_columns_by_task_name():
    dataset = helper_testing.get_micro_hafnia_dataset(dataset_name="micro-tiny-dataset", force_update=False)
    table = dataset.samples

    def check_expected_column_names(table_before, table_after, PrimitiveTypes: List[Type[Primitive]]):
        for PrimitiveType in PrimitiveTypes:
            assert PrimitiveType.column_name() not in table_after.columns
            task_names = (
                table_before[PrimitiveType.column_name()]
                .explode()
                .struct.field(PrimitiveField.TASK_NAME)
                .unique()
                .to_list()
            )
            for task_name in task_names:
                assert f"{PrimitiveType.column_name()}.{task_name}" in table_after.columns

    coordinate_types = [Classification]
    table_out = table_transformations.split_primitive_columns_by_task_name(table, coordinate_types=coordinate_types)
    check_expected_column_names(table_before=table, table_after=table_out, PrimitiveTypes=coordinate_types)

    coordinate_types = [Classification, Bbox]
    table_out = table_transformations.split_primitive_columns_by_task_name(table, coordinate_types=coordinate_types)
    check_expected_column_names(table_before=table, table_after=table_out, PrimitiveTypes=coordinate_types)

    coordinate_types = [Classification, Bbox]
    table_out = table_transformations.split_primitive_columns_by_task_name(table)
    check_expected_column_names(table_before=table, table_after=table_out, PrimitiveTypes=coordinate_types)


def test_unnest_classification_tasks():
    dataset = helper_testing.get_micro_hafnia_dataset(dataset_name="micro-tiny-dataset", force_update=False)
    table = dataset.samples

    table_unnested = unnest_classification_tasks(table)

    class_tasks = (
        table[Classification.column_name()].explode().struct.field(PrimitiveField.TASK_NAME).unique().to_list()
    )

    for task_name in class_tasks:
        expected_column_name = f"{Classification.column_name()}.{task_name}"
        assert expected_column_name in table_unnested.columns
        assert table_unnested[expected_column_name].dtype == pl.Struct


@pytest.mark.parametrize(
    "bboxes0, bboxes1, expected_fields",
    [
        # Extra field in 'samples0' only. The shared fields should be kept in the field order of 'samples0'.
        ([{"a": 1, "b": 2, "c": 3, "meta0": 9}], [{"a": 1, "b": 2, "c": 3}], ["a", "b", "c"]),
        # Extra field in 'samples1' only
        ([{"a": 1, "b": 2, "c": 3}], [{"a": 1, "b": 2, "c": 3, "meta1": 8}], ["a", "b", "c"]),
        # Extra field in both datasets
        ([{"a": 1, "b": 2, "meta0": 9}], [{"a": 1, "b": 2, "meta1": 8}], ["a", "b"]),
        # Same fields, but declared in a different order
        ([{"a": 1, "b": 2, "c": 3}], [{"c": 3, "a": 1, "b": 2}], ["a", "b", "c"]),
    ],
)
def test_merge_samples_with_non_matching_primitive_fields(bboxes0, bboxes1, expected_fields):
    """Primitive columns should survive a merge even when the struct fields do not match exactly."""
    samples0 = pl.DataFrame({SampleField.FILE_PATH: ["0.png"], Bbox.column_name(): [bboxes0]})
    samples1 = pl.DataFrame({SampleField.FILE_PATH: ["1.png"], Bbox.column_name(): [bboxes1]})

    merged = table_transformations.merge_samples(samples0, samples1)

    assert Bbox.column_name() in merged.columns, "The primitive column should not be dropped by the merge"
    merged_fields = [field.name for field in merged.schema[Bbox.column_name()].inner.fields]
    assert merged_fields == expected_fields
    assert len(merged) == 2

    # Struct fields are matched by name, so the values of both datasets should be preserved
    bboxes_merged = merged.explode(Bbox.column_name()).unnest(Bbox.column_name())
    for field_name in expected_fields:
        expected_values = [bboxes0[0][field_name], bboxes1[0][field_name]]
        assert bboxes_merged[field_name].to_list() == expected_values


def test_merge_samples_drops_primitive_column_without_shared_fields():
    samples0 = pl.DataFrame({SampleField.FILE_PATH: ["0.png"], Bbox.column_name(): [[{"a": 1}]]})
    samples1 = pl.DataFrame({SampleField.FILE_PATH: ["1.png"], Bbox.column_name(): [[{"b": 2}]]})

    merged = table_transformations.merge_samples(samples0, samples1)

    assert Bbox.column_name() not in merged.columns
    assert len(merged) == 2


def get_skeleton_dataset(dataset_name: str, template: Optional[SkeletonTemplate] = None) -> HafniaDataset:
    """Dataset with a single skeleton annotation. Edges are defined by the template of the class."""
    template = template or SkeletonTemplate(keypoint_names=["a", "b"], edges=[SkeletonEdge(index_start=0, index_end=1)])
    task = TaskInfo(primitive=Skeleton, classes=[ClassInfo(name="pose", skeleton=template)])
    keypoints = [
        KeyPoint(point=Point(x=0.1, y=0.2), class_name="a", class_idx=0),
        KeyPoint(point=Point(x=0.3, y=0.4), class_name="b", class_idx=1),
    ]
    skeleton = Skeleton(keypoints=keypoints, class_name="pose", class_idx=0)
    sample = Sample(file_path="image.jpg", height=10, width=10, split="train", skeletons=[skeleton])
    return HafniaDataset.from_samples_list([sample], info=DatasetInfo(dataset_name=dataset_name, tasks=[task]))


def test_merge_samples_keeps_nested_primitive_fields_with_non_matching_types():
    """A nested field with the same name but a different type should not drop the whole annotation."""
    keypoints0 = [{"class_name": "a", "created_at": datetime(2024, 1, 1)}]
    keypoints1 = [{"class_name": "a", "created_at": None}]
    samples0 = pl.DataFrame({SampleField.FILE_PATH: ["0.png"], Skeleton.column_name(): [[{"keypoints": keypoints0}]]})
    samples1 = pl.DataFrame({SampleField.FILE_PATH: ["1.png"], Skeleton.column_name(): [[{"keypoints": keypoints1}]]})

    merged = table_transformations.merge_samples(samples0, samples1)

    assert Skeleton.column_name() in merged.columns
    merged_fields = [field.name for field in merged.schema[Skeleton.column_name()].inner.fields]
    assert merged_fields == ["keypoints"], "Expected the nested keypoints to survive the merge"
    keypoint_names = merged[Skeleton.column_name()].explode().struct.field("keypoints").to_list()
    assert [kp[0]["class_name"] for kp in keypoint_names] == ["a", "a"]
    assert len(merged) == 2


def test_merge_datasets_with_conflicting_skeleton_templates():
    dataset0 = get_skeleton_dataset(dataset_name="dataset0")

    other_template = SkeletonTemplate(keypoint_names=["a", "b"], edges=[SkeletonEdge(index_start=1, index_end=0)])
    dataset1 = get_skeleton_dataset(dataset_name="dataset1", template=other_template)

    with pytest.raises(ValueError, match="different skeleton templates for the same class"):
        HafniaDataset.merge(dataset0, dataset1)


@pytest.mark.parametrize("skeletons", [[], None], ids=["empty_list", "no_skeletons"])
def test_dataset_without_skeleton_annotations(skeletons):
    """A skeleton task without annotations gives a 'List(Null)'/'Null' column that has no struct fields."""
    template = SkeletonTemplate(keypoint_names=["a", "b"], edges=[SkeletonEdge(index_start=0, index_end=1)])
    task = TaskInfo(primitive=Skeleton, classes=[ClassInfo(name="pose", skeleton=template)])
    sample = Sample(file_path="image.jpg", height=10, width=10, split="train", skeletons=skeletons)

    dataset = HafniaDataset.from_samples_list([sample], info=DatasetInfo(dataset_name="empty", tasks=[task]))

    assert len(dataset) == 1
