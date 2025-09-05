from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from hafnia.dataset.dataset_recipe.dataset_recipe import DatasetRecipe
from hafnia.dataset.dataset_recipe.recipe_transforms import SelectSamples, Shuffle


@dataclass
class TestUseCaseImplicit2Explicit:
    name: str
    recipe_implicit: Any
    recipe_explicit: DatasetRecipe


@pytest.mark.parametrize(
    "test_case",
    [
        TestUseCaseImplicit2Explicit(
            name="str to DatasetFromName",
            recipe_implicit="mnist",
            recipe_explicit=DatasetRecipe.from_name(name="mnist"),
        ),
        TestUseCaseImplicit2Explicit(
            name="Path to DatasetFromPath",
            recipe_implicit=Path("path/to/dataset"),
            recipe_explicit=DatasetRecipe.from_path(path_folder=Path("path/to/dataset")),
        ),
        TestUseCaseImplicit2Explicit(
            name="tuple to DatasetMerger",
            recipe_implicit=("dataset1", "dataset2"),
            recipe_explicit=DatasetRecipe.from_merger(
                recipes=[
                    DatasetRecipe.from_name(name="dataset1", force_redownload=False),
                    DatasetRecipe.from_name(name="dataset2", force_redownload=False),
                ],
            ),
        ),
        TestUseCaseImplicit2Explicit(
            name="tuple One dataset DatasetMerger",
            recipe_implicit=("dataset1",),
            recipe_explicit=DatasetRecipe.from_name(name="dataset1", force_redownload=False),
        ),
        TestUseCaseImplicit2Explicit(
            name="list to DatasetRecipe",
            recipe_implicit=["dataset1", SelectSamples(n_samples=10), Shuffle()],
            recipe_explicit=DatasetRecipe.from_name(name="dataset1").select_samples(n_samples=10).shuffle(),
        ),
        TestUseCaseImplicit2Explicit(
            name="DatasetFromName to DatasetFromName (no change)",
            recipe_implicit=DatasetRecipe.from_name(name="mnist", force_redownload=False),
            recipe_explicit=DatasetRecipe.from_name(name="mnist", force_redownload=False),
        ),
        TestUseCaseImplicit2Explicit(
            name="DatasetFromPath to DatasetFromPath (no change)",
            recipe_implicit=DatasetRecipe.from_path(path_folder=Path("path/to/dataset"), check_for_images=True),
            recipe_explicit=DatasetRecipe.from_path(path_folder=Path("path/to/dataset"), check_for_images=True),
        ),
        TestUseCaseImplicit2Explicit(
            name="DatasetMerger to DatasetMerger (no change)",
            recipe_implicit=DatasetRecipe.from_merge(
                recipe0=DatasetRecipe.from_name(name="dataset1", force_redownload=False),
                recipe1=DatasetRecipe.from_name(name="dataset2", force_redownload=False),
            ),
            recipe_explicit=DatasetRecipe.from_merge(
                recipe0=DatasetRecipe.from_name(name="dataset1", force_redownload=False),
                recipe1=DatasetRecipe.from_name(name="dataset2", force_redownload=False),
            ),
        ),
        TestUseCaseImplicit2Explicit(
            name="Transforms to Transforms (no change)",
            recipe_implicit=DatasetRecipe.from_name(name="dataset1", force_redownload=False)
            .select_samples(n_samples=10)
            .shuffle(),
            recipe_explicit=DatasetRecipe.from_name(name="dataset1", force_redownload=False)
            .select_samples(n_samples=10)
            .shuffle(),
        ),
        TestUseCaseImplicit2Explicit(
            name="Mix implicit/explicit recipes",
            recipe_implicit=(
                DatasetRecipe.from_name(name="dataset1", force_redownload=False),
                Path("path/to/dataset"),
                ["dataset2", SelectSamples(n_samples=5), Shuffle()],
                DatasetRecipe.from_name(name="dataset2", force_redownload=False).select_samples(n_samples=5).shuffle(),
                ("dataset2", DatasetRecipe.from_name(name="dataset3", force_redownload=False)),
                "dataset4",
            ),
            recipe_explicit=DatasetRecipe.from_merger(
                recipes=[
                    DatasetRecipe.from_name(name="dataset1", force_redownload=False),
                    DatasetRecipe.from_path(path_folder=Path("path/to/dataset"), check_for_images=True),
                    DatasetRecipe.from_name(name="dataset2").select_samples(n_samples=5).shuffle(),
                    DatasetRecipe.from_name(name="dataset2", force_redownload=False)
                    .select_samples(n_samples=5)
                    .shuffle(),
                    DatasetRecipe.from_merger(
                        recipes=[
                            DatasetRecipe.from_name(name="dataset2", force_redownload=False),
                            DatasetRecipe.from_name(name="dataset3", force_redownload=False),
                        ],
                    ),
                    DatasetRecipe.from_name(name="dataset4", force_redownload=False),
                ],
            ),
        ),
    ],
    ids=lambda test_case: test_case.name,  # To use the name of the test case as the ID for clarity
)
def test_cases_implicit_to_explicit_conversion(test_case: TestUseCaseImplicit2Explicit):
    actual_recipe = DatasetRecipe.from_implicit_form(test_case.recipe_implicit)

    assert isinstance(actual_recipe, DatasetRecipe)  # type: ignore
    assert actual_recipe == test_case.recipe_explicit
    import rich

    rich.print(actual_recipe)
    rich.print(test_case.recipe_explicit)
