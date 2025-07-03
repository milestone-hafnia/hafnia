from pathlib import Path

import pytest

from hafnia.dataset.data_recipe.data_recipes import (
    DataRecipe,
    DatasetRecipeFromName,
    DatasetRecipeFromPath,
    RecipeMerger,
    RecipeTransform,
    RecipeTransforms,
    Serializable,
)
from hafnia.dataset.data_recipe.recipe_transformations import SelectSamples, Shuffle, SplitsByRatios
from hafnia.dataset.hafnia_dataset import HafniaDataset
from hafnia.dataset.operations import dataset_transformations
from hafnia.helper_testing import get_hafnia_functions_from_module, is_hafnia_configured
from hafnia.utils import snake_to_pascal_case


def get_data_recipe() -> DataRecipe:
    dataset_recipe = RecipeMerger(
        recipes=[
            RecipeTransforms(
                recipe=DatasetRecipeFromName(name="mnist", force_redownload=False),
                transforms=[
                    SelectSamples(n_samples=20, shuffle=True, seed=42),
                    Shuffle(seed=123),
                ],
            ),
            RecipeTransforms(
                recipe=DatasetRecipeFromName(name="mnist", force_redownload=False),
                transforms=[
                    SelectSamples(n_samples=30, shuffle=True, seed=42),
                    SplitsByRatios(split_ratios={"train": 0.8, "val": 0.1, "test": 0.1}, seed=42),
                ],
            ),
            DatasetRecipeFromName(name="mnist", force_redownload=False),
        ]
    )

    return dataset_recipe


@pytest.mark.parametrize("recipe_transform", RecipeTransform.get_nested_subclasses())
def test_serializable_functions_check_signature(recipe_transform: RecipeTransform):
    """
    RecipeTransform converts a function into a serializable class.
    It ensures that the function signature is the same as the expected model fields.
    """
    recipe_transform.check_signature()  # Ensures that function signatures match the expected model fields


@pytest.mark.parametrize("transformation_function_name", get_hafnia_functions_from_module(dataset_transformations))
def test_check_dataset_transformations_have_builders(transformation_function_name: str):
    """
    Ensure that all dataset transformations have a corresponding RecipeTransform.
    """
    skip_list = ["transform_images", "merge"]
    if transformation_function_name in skip_list:
        pytest.skip(f"Skipping {transformation_function_name} as it is not a RecipeTransform.")

    function_name_pascal_case = snake_to_pascal_case(transformation_function_name)
    serializable_functions = list(RecipeTransform.name_to_type_mapping())

    expected_class_name = f"class {function_name_pascal_case}({RecipeTransform.__name__}):"
    assert function_name_pascal_case in serializable_functions, (
        f"Transformation function '{transformation_function_name}' in 'operations/dataset_transformations.py' "
        "does not have a corresponding '{expected_class_name}' in 'data_recipe/recipe_transformations.py.py'. \n"
        " We expect all functions in 'operations/dataset_transformations.py' to have a 'corresponding' class in "
        f"'data_recipe/recipe_transformations.py.py'.\n"
        f"Please add '{expected_class_name}' class for it in 'data_recipe/recipe_transformations.py.py'."
    )


def test_dataset_recipe_serialization_deserialization_dict():
    """
    Test that Serializable can be serialized and deserialized correctly.
    """
    dataset_recipe = get_data_recipe()

    # Smoke test - it can be serialized
    serialized_data: dict = dataset_recipe.model_dump()  # type: ignore[annotation-unchecked]

    # Smoke test - it can be deserialized back
    deserialized_recipe = Serializable.from_dict(serialized_data)

    assert isinstance(deserialized_recipe, DataRecipe)  # type: ignore[misc]
    assert deserialized_recipe == dataset_recipe, "Deserialized recipe does not match original recipe"


def test_dataset_recipe_serialization_deserialization_json():
    """
    Test that Serializable can be serialized and deserialized correctly.
    """
    dataset_recipe = get_data_recipe()

    # Smoke test - it can be serialized
    serialized_data: str = dataset_recipe.model_dump_json()

    # Smoke test - it can be deserialized
    deserialized_recipe = Serializable.from_json_str(serialized_data)

    assert isinstance(deserialized_recipe, DataRecipe)  # type: ignore[misc]
    assert deserialized_recipe == dataset_recipe, "Deserialized recipe does not match original recipe"


@pytest.mark.parametrize(
    "dataset_recipe",
    [
        DatasetRecipeFromName(name="mnist", force_redownload=False),
        DatasetRecipeFromPath(path_folder=Path(".data/datasets/mnist"), check_for_images=False),
        RecipeMerger(
            recipes=[
                DatasetRecipeFromName(name="mnist", force_redownload=False),
                DatasetRecipeFromPath(path_folder=Path(".data/datasets/mnist"), check_for_images=False),
            ]
        ),
        RecipeTransforms(
            recipe=DatasetRecipeFromName(name="mnist", force_redownload=False),
            transforms=[
                SelectSamples(n_samples=20, shuffle=True, seed=42),
                Shuffle(seed=123),
            ],
        ),
    ],
)
def test_build_dataset_recipe(dataset_recipe: DataRecipe):
    """
    Test that LoadDataset recipe can be created and serialized.
    """
    if not is_hafnia_configured():
        pytest.skip("Hafnia is not configured, skipping build dataset recipe tests.")

    # Checks that the dataset recipe can be called and returns as a HafniaDataset
    dataset: HafniaDataset = dataset_recipe.build()

    # Checks that complex dataset recipes can be built, stored and loaded
    path_dataset = HafniaDataset.from_recipe_to_disk(dataset_recipe)
    assert isinstance(path_dataset, Path), "Path dataset is not an instance of Path"
    dataset = HafniaDataset.from_path(path_dataset, check_for_images=False)
    assert isinstance(dataset, HafniaDataset), "Dataset is not an instance of HafniaDataset"
    # assert isinstance(dataset, HafniaDataset), "Dataset is not an instance of HafniaDataset"
