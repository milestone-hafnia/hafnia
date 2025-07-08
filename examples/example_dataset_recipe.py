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

### Introducing DataRecipe ###
# A DataRecipe is a blueprint for the dataset you want to create.
# The recipe itself is not executed - this is just a specification of the dataset you want!

# Dataset recipe from name
dataset_recipe: DatasetRecipe = DatasetRecipe.from_name(name="mnist")


# Dataset recipe from path
dataset_recipe: DatasetRecipe = DatasetRecipe.from_path(path_folder=Path(".data/datasets/mnist"))

# Merge recipes into one recipe
dataset_recipe: DatasetRecipe = DatasetRecipe.from_merger(
    recipes=[
        DatasetRecipe.from_name(name="mnist"),
        DatasetRecipe.from_name(name="mnist"),
    ]
)

# Recipe with transformations
dataset_recipe: DatasetRecipe = (
    DatasetRecipe.from_name(name="mnist").select_samples(n_samples=20, shuffle=True, seed=42).shuffle(seed=123)
)

rprint(dataset_recipe.as_json_str())

# To actually generate the dataset, you call build() on the recipe.
merged_dataset: HafniaDataset = dataset_recipe.build()

# Or use the `load_dataset` function to load the dataset directly.
merged_dataset: HafniaDataset = load_dataset(dataset_recipe)
# You get a few extra things when using `load_dataset`.
# 1) You can use an implicit form of the recipe (described later).
# 2) The dataset is cached if it exists, so you don't have to
#    download or rebuild the dataset every time.

assert len(merged_dataset) == 20

# Recipes can be infinitely nested and combined.
# This includes: loading, sampling, shuffling, splitting, and merging datasets.
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
assert len(dataset) == 450  # 20 + 30 + 2x200

# An important feature of the dataset recipe is that it can be serialized to JSON
# and deserialize back to the original recipe. Meaning that the dataset recipe can be saved, shared,
# loaded and built as a dataset.
#
# This allows you to do the following:
# 1) For model training, a desired dataset can be defined from a configuration file - without changing the code.
#    Improving configurability and reproducibility of your experiments.
# 2) The data recipe is also useful for Training as a Service (TaaS) as it allows you to define the dataset
#    you want in a configuration file and load it in the TaaS platform.
# 3) Finally, creating a recipe file requires you to define


path_json = Path(".data/tmp/dataset_recipe.json")
dataset_recipe.as_json_file(path_json)

# And load the dataset recipe from a file
dataset_recipe_again = DatasetRecipe.from_json_file(path_json)

assert dataset_recipe_again == dataset_recipe


## Data recipe in implicit form
# Above recipe becomes very verbose for many operations. To simplify this, you can use an implicit form.
# The implicit form allows you to specify datasets in a more concise way using below rules:
#    str: Will get a dataset by name -> In explicit form it becomes 'DatasetRecipe.from_name'
#    Path: Will get a dataset from path -> In explicit form it becomes 'DatasetRecipe.from_path'
#    tuple: Will merge datasets specified in the tuple -> In explicit form it becomes 'DatasetRecipe.from_merger'
#    list: Will define a dataset followed by a list of transformations -> In explicit form it becomes 'DatasetRecipe()'

split_ratio = {"train": 0.8, "val": 0.1, "test": 0.1}
implicit_recipe = (
    ("mnist", "mnist"),
    [Path(".data/datasets/mnist"), SelectSamples(n_samples=30), SplitsByRatios(split_ratios=split_ratio)],
    ["mnist", SelectSamples(n_samples=20), Shuffle()],
)


# Test the conversion function
explicit_recipe = DatasetRecipe.from_implicit_form(implicit_recipe)
rprint("Converted explicit recipe:")
rprint(explicit_recipe)

assert explicit_recipe == dataset_recipe_again
