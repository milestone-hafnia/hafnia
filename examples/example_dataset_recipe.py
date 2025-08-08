from pathlib import Path

from rich import print as rprint

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

# A DatasetRecipe is an important concept in Hafnia as it allows you to merge multiple datasets
# and transformations in a single recipe. This is especially useful for Training as a Service (TaaS)
# where you need to define the dataset you want as a configuration and load it in the TaaS platform.

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

# Or as a JSON string:
json_str: str = dataset_recipe.as_json_str()
rprint(json_str)

# This is an important feature of a 'DatasetRecipe' it only registers operations and that the recipe itself
# - and not the dataset - can be saved as a file and loaded from file.
# Meaning you can easily save, share, load and build the dataset later or in a different environment.
# For TaaS, this is the only way to include multiple datasets during training.


# 2) The recipe can be loaded from json string
dataset_recipe_again: DatasetRecipe = DatasetRecipe.from_json_str(json_str)
# dataset_recipe_again.build()

# We can verify that the loaded recipe is the same as the original recipe.
assert dataset_recipe_again == dataset_recipe

# Additionally, you can get the python code for creating the same recipe.
dataset_recipe.as_python_code()

# Example: DatasetRecipe from Path
dataset_recipe = DatasetRecipe.from_path(path_folder=Path(".data/datasets/mnist"))

# Example: DatasetRecipe by merging multiple dataset recipes
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
rprint("✓ Conversion successful - recipes are equivalent!")
