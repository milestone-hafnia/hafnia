from pathlib import Path
from typing import Any, Optional, Union

from hafnia import utils
from hafnia.dataset.data_recipe.data_recipes import (
    DataRecipe,
    DatasetRecipeFromName,
    DatasetRecipeFromPath,
    RecipeMerger,
    RecipeTransforms,
)


def convert_to_explicit_recipe_form(recipe: Any) -> DataRecipe:
    """
    Recursively convert from implicit recipe to explicit form.
    Handles mixed implicit/explicit recipes.

    Conversion rules:
    - str: Will get a dataset by name -> DatasetRecipeFromName
    - Path: Will get a dataset from path -> DatasetRecipeFromPath
    - tuple: Will merge datasets specified in the tuple -> RecipeMerger
    - list: Will define a list of transformations -> RecipeTransforms

    Example: DataRecipe from dataset name:
    ```python
    recipe_implicit = "mnist"
    recipe_explicit = convert_to_explicit_recipe_form(recipe_implicit)
    >>> recipe_explicit
    DatasetRecipeFromName(dataset_name='mnist', force_redownload=False)
    ```

    Example: DataRecipe from tuple (merging multiple recipes):
    ```python
    recipe_implicit = ("dataset1", "dataset2")
    recipe_explicit = convert_to_explicit_recipe_form(recipe_implicit)
    >>> recipe_explicit
    RecipeMerger(
        recipes=[
            DatasetRecipeFromName(dataset_name='dataset1', force_redownload=False),
            DatasetRecipeFromName(dataset_name='dataset2', force_redownload=False)
        ]
    )

    Example: DataRecipe from list (recipe and transformations):
    ```python
    recipe_implicit = ["mnist", SelectSamples(n_samples=20), Shuffle(seed=123)]
    recipe_explicit = convert_to_explicit_recipe_form(recipe_implicit)
    >>> recipe_explicit
    Transforms(
        recipe=DatasetRecipeFromName(dataset_name='mnist', force_redownload=False),
        transforms=[SelectSamples(n_samples=20), Shuffle(seed=123)]
    )
    ```

    """
    if isinstance(recipe, DataRecipe):  # type: ignore
        # It is possible to do an early return if recipe is a 'DataRecipe'-type even for nested and
        # potentially mixed recipes. If you (really) think about it, this might surprise you,
        # as this will bypass the conversion logic for nested recipes.
        # However, this is not a problem as 'DataRecipe' classes are also pydantic models,
        # so if a user introduces a 'DataRecipe'-class in the recipe (in potentially
        # some nested and mixed implicit/explicit form) it will (due to pydantic validation) force
        # the user to specify all nested recipes to be converted to explicit form.
        return recipe

    if isinstance(recipe, str):  # str-type is convert to DatasetFromName
        return DatasetRecipeFromName(name=recipe)

    if isinstance(recipe, Path):  # Path-type is convert to DatasetFromPath
        return DatasetRecipeFromPath(path_folder=recipe)

    if isinstance(recipe, tuple):  # tuple-type is convert to DatasetMerger
        recipes = [convert_to_explicit_recipe_form(item) for item in recipe]
        return RecipeMerger(recipes=recipes)

    if isinstance(recipe, list):  # list-type is convert to Transforms
        if len(recipe) == 0:
            raise ValueError("List of recipes cannot be empty")

        dataset_recipe = recipe[0]  # First element is the dataset recipe
        loader = convert_to_explicit_recipe_form(dataset_recipe)

        transforms = recipe[1:]  # Remaining items are transformations
        return RecipeTransforms(recipe=loader, transforms=transforms)

    raise ValueError(f"Unsupported recipe type: {type(recipe)}")


def unique_name_from_recipe(recipe: DataRecipe) -> str:
    if isinstance(recipe, DatasetRecipeFromName):
        # If the dataset recipe is simply a DatasetFromName, we bypass the hashing logic
        # and return the name directly. The dataset is already uniquely identified by its name.
        # Add  version if need... Optionally, you may also completely delete this exception
        # and always return the unique name including the hash to support versioning.
        return recipe.name
    recipe_json_str = recipe.model_dump_json()
    hash_recipe = utils.hash_from_string(recipe_json_str)
    short_recipe_str = recipe.short_name()
    unique_name = f"{short_recipe_str}_{hash_recipe}"
    return unique_name


def get_dataset_path_from_recipe(recipe: DataRecipe, path_datasets: Optional[Union[Path, str]] = None) -> Path:
    path_datasets = path_datasets or utils.PATH_DATASETS
    path_datasets = Path(path_datasets)
    unique_dataset_name = unique_name_from_recipe(recipe)
    return path_datasets / unique_dataset_name
