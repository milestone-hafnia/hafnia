from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from hafnia.dataset.builder.builder_helpers import convert_to_explicit_specification
from hafnia.dataset.builder.builders import DatasetBuilder, DatasetFromName, DatasetFromPath, DatasetMerger, Transforms
from hafnia.dataset.builder.dataset_transformations import Sample, Shuffle


@dataclass
class TestUseCaseImplicit2Explicit:
    name: str
    implicit_specification: Any
    expected_explicit_specification: DatasetBuilder

    def __str__(self):
        return self.name


@pytest.mark.parametrize(
    "test_case",
    [
        TestUseCaseImplicit2Explicit(
            name="str to DatasetFromName",
            implicit_specification="mnist",
            expected_explicit_specification=DatasetFromName(name="mnist", force_redownload=False),
        ),
        TestUseCaseImplicit2Explicit(
            name="Path to DatasetFromPath",
            implicit_specification=Path("path/to/dataset"),
            expected_explicit_specification=DatasetFromPath(path_folder=Path("path/to/dataset"), check_for_images=True),
        ),
        TestUseCaseImplicit2Explicit(
            name="tuple to DatasetMerger",
            implicit_specification=("dataset1", "dataset2"),
            expected_explicit_specification=DatasetMerger(
                builders=[
                    DatasetFromName(name="dataset1", force_redownload=False),
                    DatasetFromName(name="dataset2", force_redownload=False),
                ]
            ),
        ),
        TestUseCaseImplicit2Explicit(
            name="list to Transforms",
            implicit_specification=["dataset1", Sample(n_samples=10), Shuffle()],
            expected_explicit_specification=Transforms(
                loader=DatasetFromName(name="dataset1", force_redownload=False),
                transforms=[Sample(n_samples=10), Shuffle()],
            ),
        ),
        TestUseCaseImplicit2Explicit(
            name="DatasetFromName to DatasetFromName (no change)",
            implicit_specification=DatasetFromName(name="mnist", force_redownload=False),
            expected_explicit_specification=DatasetFromName(name="mnist", force_redownload=False),
        ),
        TestUseCaseImplicit2Explicit(
            name="DatasetFromPath to DatasetFromPath (no change)",
            implicit_specification=DatasetFromPath(path_folder=Path("path/to/dataset"), check_for_images=True),
            expected_explicit_specification=DatasetFromPath(path_folder=Path("path/to/dataset"), check_for_images=True),
        ),
        TestUseCaseImplicit2Explicit(
            name="DatasetMerger to DatasetMerger (no change)",
            implicit_specification=DatasetMerger(
                builders=[
                    DatasetFromName(name="dataset1", force_redownload=False),
                    DatasetFromName(name="dataset2", force_redownload=False),
                ]
            ),
            expected_explicit_specification=DatasetMerger(
                builders=[
                    DatasetFromName(name="dataset1", force_redownload=False),
                    DatasetFromName(name="dataset2", force_redownload=False),
                ]
            ),
        ),
        TestUseCaseImplicit2Explicit(
            name="Transforms to Transforms (no change)",
            implicit_specification=Transforms(
                loader=DatasetFromName(name="dataset1", force_redownload=False),
                transforms=[Sample(n_samples=10), Shuffle()],
            ),
            expected_explicit_specification=Transforms(
                loader=DatasetFromName(name="dataset1", force_redownload=False),
                transforms=[Sample(n_samples=10), Shuffle()],
            ),
        ),
        TestUseCaseImplicit2Explicit(
            name="Mix implicit/explicit specifications",
            implicit_specification=(
                DatasetFromName(name="dataset1", force_redownload=False),
                Path("path/to/dataset"),
                ["dataset2", Sample(n_samples=5), Shuffle()],
                Transforms(
                    loader=DatasetFromName(name="dataset2", force_redownload=False),
                    transforms=[Sample(n_samples=5), Shuffle()],
                ),
                ("dataset2", DatasetFromName(name="dataset3", force_redownload=False)),
                "dataset4",
            ),
            expected_explicit_specification=DatasetMerger(
                builders=[
                    DatasetFromName(name="dataset1", force_redownload=False),
                    DatasetFromPath(path_folder=Path("path/to/dataset"), check_for_images=True),
                    Transforms(
                        loader=DatasetFromName(name="dataset2", force_redownload=False),
                        transforms=[Sample(n_samples=5), Shuffle()],
                    ),
                    Transforms(
                        loader=DatasetFromName(name="dataset2", force_redownload=False),
                        transforms=[Sample(n_samples=5), Shuffle()],
                    ),
                    DatasetMerger(
                        builders=[
                            DatasetFromName(name="dataset2", force_redownload=False),
                            DatasetFromName(name="dataset3", force_redownload=False),
                        ]
                    ),
                    DatasetFromName(name="dataset4", force_redownload=False),
                ],
            ),
        ),
    ],
    ids=lambda test_case: test_case.name,  # To use the name of the test case as the ID for clarity
)
def test_implicit_to_explicit_conversion(test_case: TestUseCaseImplicit2Explicit):
    actual_specification = convert_to_explicit_specification(test_case.implicit_specification)

    assert isinstance(actual_specification, DatasetBuilder)  # type: ignore
    assert actual_specification == test_case.expected_explicit_specification
