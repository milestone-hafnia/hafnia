from pathlib import Path

from rich import print as rprint

from hafnia import utils
from hafnia.data.factory import load_dataset
from hafnia.dataset.dataset_recipe.dataset_recipe import DatasetRecipe
from hafnia.dataset.dataset_recipe.recipe_transforms import (
    SelectSamples,
    Shuffle,
    SplitsByRatios,
)
from hafnia.dataset.hafnia_dataset import HafniaDataset

### Introducing DatasetRecipe ###
# A DatasetRecipe is a recipe for the dataset you want to create.
# The recipe itself is not executed - this is just a specification of the dataset you want!

# The 'DatasetRecipe' interface is similar to the 'HafniaDataset' interface.
# To demonstrate, we will first create a dataset with the regular 'HafniaDataset' interface.
# This line will get the "mnist" dataset, shuffle it, and select 20 samples.
dataset = HafniaDataset.from_name(name="mnist").shuffle().select_samples(n_samples=20)

# Now the same dataset is created using the 'DatasetRecipe' interface.
dataset_recipe = DatasetRecipe.from_name(name="mnist").shuffle().select_samples(n_samples=20)
dataset = dataset_recipe.build()
# Note that the interface is similar, but to actually create the dataset you need to call `build()` on the recipe.

# Unlike the HafniaDataset, a DatasetRecipe does not execute operations. It only registers
# the operations applied to the recipe and can be used to build the dataset later.
# You can print the dataset recipe to the operations that were applied to it.
rprint(dataset_recipe)

# The key for recipes is that they can be saved and loaded as a JSON.
# This also allows the recipe to be saved, shared, loaded and used later to build a dataset
# in a different environment.

# Example: Saving and loading a dataset recipe from file.
path_recipe = Path(".data/dataset_recipes/example_recipe.json")
json_str: str = dataset_recipe.as_json_file(path_recipe)
dataset_recipe_again: DatasetRecipe = DatasetRecipe.from_json_file(path_recipe)

# Verify that the loaded recipe is identical to the original recipe.
assert dataset_recipe_again == dataset_recipe

# It is also possible to generate the recipe as python code
dataset_recipe.as_python_code()

# The recipe also allows you to combine multiple datasets and transformations that can be
# executed in the TaaS platform. This is demonstrated below:
if utils.is_hafnia_configured():  # First ensure you are connected to the hafnia platform
    # Upload the dataset recipe - this will make it available for TaaS and for users of your organization
    dataset_recipe.as_platform_recipe(recipe_name="example-mnist-recipe")

    # The recipe is now available in TaaS, for different environments and other users in your organization
    dataset_recipe_again = DatasetRecipe.from_recipe_name(name="example-mnist-recipe")

    # Launch an experiment with the dataset recipe using the CLI:
    # hafnia experiment create --dataset-recipe example-mnist-recipe --trainer-path ../trainer-classification

    # Coming soon: Dataset recipes will be included in the web platform to them to be shared, managed
    # and used in experiments.

### More examples dataset recipes ###
# Example: 'DatasetRecipe' by merging multiple dataset recipes
dataset_recipe = DatasetRecipe.from_merger(
    recipes=[
        DatasetRecipe.from_name(name="mnist"),
        DatasetRecipe.from_name(name="mnist"),
    ]
)

# Example: Recipes can be infinitely nested and combined.
dataset_recipe = DatasetRecipe.from_merger(
    recipes=[
        DatasetRecipe.from_merger(
            recipes=[
                DatasetRecipe.from_name(name="mnist"),
                DatasetRecipe.from_name(name="mnist"),
            ]
        ),
        DatasetRecipe.from_path(path_folder=Path(".data/datasets/mnist"))
        .select_samples(n_samples=30)
        .splits_by_ratios(split_ratios={"train": 0.8, "val": 0.1, "test": 0.1}),
        DatasetRecipe.from_name(name="mnist").select_samples(n_samples=20).shuffle(),
    ]
)

# Now you can build the dataset from the recipe.
dataset: HafniaDataset = dataset_recipe.build()
assert len(dataset) == 450  # 2x200 + 30 + 20

# Finally, you can print the dataset recipe to see what it contains.
rprint(dataset_recipe)  # as a python object
print(dataset_recipe.as_json_str())  # as a JSON string


# Example: Using the 'load_dataset' function
merged_dataset: HafniaDataset = load_dataset(dataset_recipe)
# You get a few extra things when using `load_dataset`.
# 1) You get the dataset directly - you don't have to call `build()` on the recipe.
# 2) The dataset is cached if it already exists, so you don't have to
#    download or rebuild the dataset on the second run.
# 3) You can use an implicit form of the recipe. One example of this is that you just specify
#    the dataset name `load_dataset("mnist")` or path `load_dataset(Path(".data/datasets/mnist"))`


### DatasetRecipe Implicit Form ###
# Below we demonstrate the difference between implicit and explicit forms of dataset recipes.
# Example: Get dataset by name with implicit and explicit forms
dataset = load_dataset("mnist")  # Implicit form
dataset = load_dataset(DatasetRecipe.from_name(name="mnist"))  # Explicit form

# Example: Get dataset from path with implicit and explicit forms:
dataset = load_dataset(Path(".data/datasets/mnist"))  # Implicit form
dataset = load_dataset(DatasetRecipe.from_path(path_folder=Path(".data/datasets/mnist")))  # Explicit form

# Example: Merge datasets with implicit and explicit forms
dataset = load_dataset(("mnist", "mnist"))  # Implicit form
dataset = load_dataset(  # Explicit form
    DatasetRecipe.from_merger(
        recipes=[
            DatasetRecipe.from_name(name="mnist"),
            DatasetRecipe.from_name(name="mnist"),
        ]
    )
)

# Example: Define a dataset with transformations using implicit and explicit forms
dataset = load_dataset(["mnist", SelectSamples(n_samples=20), Shuffle()])  # Implicit form
dataset = load_dataset(DatasetRecipe.from_name(name="mnist").select_samples(n_samples=20).shuffle())  # Explicit form


# Example: Complex nested example with implicit vs explicit forms
# Implicit form of a complex dataset recipe
split_ratio = {"train": 0.8, "val": 0.1, "test": 0.1}
implicit_recipe = (
    ("mnist", "mnist"),
    [Path(".data/datasets/mnist"), SelectSamples(n_samples=30), SplitsByRatios(split_ratios=split_ratio)],
    ["mnist", SelectSamples(n_samples=20), Shuffle()],
)

# Explicit form of the same complex dataset recipe
explicit_recipe = DatasetRecipe.from_merger(
    recipes=[
        DatasetRecipe.from_merger(
            recipes=[
                DatasetRecipe.from_name(name="mnist"),
                DatasetRecipe.from_name(name="mnist"),
            ]
        ),
        DatasetRecipe.from_path(path_folder=Path(".data/datasets/mnist"))
        .select_samples(n_samples=30)
        .splits_by_ratios(split_ratios=split_ratio),
        DatasetRecipe.from_name(name="mnist").select_samples(n_samples=20).shuffle(),
    ]
)

# The implicit form uses the following rules:
#    str: Will get a dataset by name -> In explicit form it becomes 'DatasetRecipe.from_name'
#    Path: Will get a dataset from path -> In explicit form it becomes 'DatasetRecipe.from_path'
#    tuple: Will merge datasets specified in the tuple -> In explicit form it becomes 'DatasetRecipe.from_merger'
#    list: Will define a dataset followed by a list of transformations -> In explicit form it becomes chained method calls
# Generally, we recommend using the explicit form over the implicit form when multiple datasets and transformations are involved.


# To convert from implicit to explicit recipe form, you can use the `from_implicit_form` method.
explicit_recipe_from_implicit = DatasetRecipe.from_implicit_form(implicit_recipe)
rprint("Converted explicit recipe:")
rprint(explicit_recipe_from_implicit)

# Verify that the conversion produces the same result
assert explicit_recipe_from_implicit == explicit_recipe
rprint("Conversion successful - recipes are equivalent!")
