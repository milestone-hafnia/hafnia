from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from hafnia.dataset.data_recipe.data_recipe_helpers import convert_to_explicit_recipe_form
from hafnia.dataset.data_recipe.data_recipes import (
    DataRecipe,
    DatasetRecipeFromName,
    DatasetRecipeFromPath,
    RecipeMerger,
    RecipeTransforms,
)
from hafnia.dataset.data_recipe.recipe_transformations import SelectSamples, Shuffle


@dataclass
class TestUseCaseImplicit2Explicit:
    name: str
    recipe_implicit: Any
    expected_recipe_explicit: DataRecipe


@pytest.mark.parametrize(
    "test_case",
    [
        TestUseCaseImplicit2Explicit(
            name="str to DatasetFromName",
            recipe_implicit="mnist",
            expected_recipe_explicit=DatasetRecipeFromName(name="mnist", force_redownload=False),
        ),
        TestUseCaseImplicit2Explicit(
            name="Path to DatasetFromPath",
            recipe_implicit=Path("path/to/dataset"),
            expected_recipe_explicit=DatasetRecipeFromPath(path_folder=Path("path/to/dataset"), check_for_images=True),
        ),
        TestUseCaseImplicit2Explicit(
            name="tuple to DatasetMerger",
            recipe_implicit=("dataset1", "dataset2"),
            expected_recipe_explicit=RecipeMerger(
                recipes=[
                    DatasetRecipeFromName(name="dataset1", force_redownload=False),
                    DatasetRecipeFromName(name="dataset2", force_redownload=False),
                ]
            ),
        ),
        TestUseCaseImplicit2Explicit(
            name="list to Transforms",
            recipe_implicit=["dataset1", SelectSamples(n_samples=10), Shuffle()],
            expected_recipe_explicit=RecipeTransforms(
                recipe=DatasetRecipeFromName(name="dataset1", force_redownload=False),
                transforms=[SelectSamples(n_samples=10), Shuffle()],
            ),
        ),
        TestUseCaseImplicit2Explicit(
            name="DatasetFromName to DatasetFromName (no change)",
            recipe_implicit=DatasetRecipeFromName(name="mnist", force_redownload=False),
            expected_recipe_explicit=DatasetRecipeFromName(name="mnist", force_redownload=False),
        ),
        TestUseCaseImplicit2Explicit(
            name="DatasetFromPath to DatasetFromPath (no change)",
            recipe_implicit=DatasetRecipeFromPath(path_folder=Path("path/to/dataset"), check_for_images=True),
            expected_recipe_explicit=DatasetRecipeFromPath(path_folder=Path("path/to/dataset"), check_for_images=True),
        ),
        TestUseCaseImplicit2Explicit(
            name="DatasetMerger to DatasetMerger (no change)",
            recipe_implicit=RecipeMerger(
                recipes=[
                    DatasetRecipeFromName(name="dataset1", force_redownload=False),
                    DatasetRecipeFromName(name="dataset2", force_redownload=False),
                ]
            ),
            expected_recipe_explicit=RecipeMerger(
                recipes=[
                    DatasetRecipeFromName(name="dataset1", force_redownload=False),
                    DatasetRecipeFromName(name="dataset2", force_redownload=False),
                ]
            ),
        ),
        TestUseCaseImplicit2Explicit(
            name="Transforms to Transforms (no change)",
            recipe_implicit=RecipeTransforms(
                recipe=DatasetRecipeFromName(name="dataset1", force_redownload=False),
                transforms=[SelectSamples(n_samples=10), Shuffle()],
            ),
            expected_recipe_explicit=RecipeTransforms(
                recipe=DatasetRecipeFromName(name="dataset1", force_redownload=False),
                transforms=[SelectSamples(n_samples=10), Shuffle()],
            ),
        ),
        TestUseCaseImplicit2Explicit(
            name="Mix implicit/explicit recipes",
            recipe_implicit=(
                DatasetRecipeFromName(name="dataset1", force_redownload=False),
                Path("path/to/dataset"),
                ["dataset2", SelectSamples(n_samples=5), Shuffle()],
                RecipeTransforms(
                    recipe=DatasetRecipeFromName(name="dataset2", force_redownload=False),
                    transforms=[SelectSamples(n_samples=5), Shuffle()],
                ),
                ("dataset2", DatasetRecipeFromName(name="dataset3", force_redownload=False)),
                "dataset4",
            ),
            expected_recipe_explicit=RecipeMerger(
                recipes=[
                    DatasetRecipeFromName(name="dataset1", force_redownload=False),
                    DatasetRecipeFromPath(path_folder=Path("path/to/dataset"), check_for_images=True),
                    RecipeTransforms(
                        recipe=DatasetRecipeFromName(name="dataset2", force_redownload=False),
                        transforms=[SelectSamples(n_samples=5), Shuffle()],
                    ),
                    RecipeTransforms(
                        recipe=DatasetRecipeFromName(name="dataset2", force_redownload=False),
                        transforms=[SelectSamples(n_samples=5), Shuffle()],
                    ),
                    RecipeMerger(
                        recipes=[
                            DatasetRecipeFromName(name="dataset2", force_redownload=False),
                            DatasetRecipeFromName(name="dataset3", force_redownload=False),
                        ]
                    ),
                    DatasetRecipeFromName(name="dataset4", force_redownload=False),
                ],
            ),
        ),
    ],
    ids=lambda test_case: test_case.name,  # To use the name of the test case as the ID for clarity
)
def test_implicit_to_explicit_conversion(test_case: TestUseCaseImplicit2Explicit):
    actual_recipe = convert_to_explicit_recipe_form(test_case.recipe_implicit)

    assert isinstance(actual_recipe, DataRecipe)  # type: ignore
    assert actual_recipe == test_case.expected_recipe_explicit
