from pathlib import Path

from rich import print as rprint

from hafnia.dataset.builder.builder_helpers import convert_to_explicit_specification
from hafnia.dataset.builder.builders import (
    DatasetBuilder,
    DatasetFromName,
    DatasetFromPath,
    DatasetMerger,
    Serializable,
    Transforms,
)
from hafnia.dataset.builder.dataset_transformations import SelectSamples, Shuffle, SplitsByRatios
from hafnia.dataset.hafnia_dataset import HafniaDataset

# Define the dataset you want with a specification
# Nothing is executed - this is just a specification of the dataset you want!
# In this example we merge two MNIST datasets, each with 200 samples.
dataset_specification: DatasetBuilder = DatasetMerger(
    builders=[
        DatasetFromName(name="mnist"),
        DatasetFromName(name="mnist"),
    ]
)

rprint(dataset_specification)

# To actually generate the dataset, you call the dataset specification
merged_dataset: HafniaDataset = dataset_specification()
assert len(merged_dataset) == 400

# You can combine multiple operations in a single dataset specification.
# This includes: loading, sampling, shuffling, splitting, and merging datasets.
dataset_specification = DatasetMerger(
    builders=[
        Transforms(
            loader=DatasetFromName(name="mnist"),
            transforms=[
                SelectSamples(n_samples=20),
                Shuffle(seed=123),
            ],
        ),
        Transforms(
            loader=DatasetFromPath(path_folder=Path(".data/datasets/mnist")),
            transforms=[
                SelectSamples(n_samples=30),
                SplitsByRatios(split_ratios={"train": 0.8, "val": 0.1, "test": 0.1}),
            ],
        ),
        DatasetMerger(
            builders=[
                DatasetFromName(name="mnist"),
                DatasetFromName(name="mnist"),
            ]
        ),
    ]
)


# To actually generate the dataset, you call the dataset specification
dataset: HafniaDataset = dataset_specification()
assert len(dataset) == 450  # 20 + 30 + 2x200

# The main reason to use the dataset specification for serialization / deserialization.
# You can save the dataset specification to a file and submit it to the hafnia platform
json_str = dataset_specification.model_dump_json()

path_json = Path(".data/tmp/dataset_specification.json")
path_json.write_text(json_str)

# You can load the dataset specification from a file
dataset_specification_again = Serializable.from_json_str(path_json.read_text())

assert dataset_specification_again == dataset_specification


## Dataset specification: Implicit form
# Above specification becomes very verbose for many operations. To simplify this, you can use an implicit form.
# The implicit form allows you to specify datasets in a more concise way using below rules:
#    str: Will get a dataset by name -> DatasetFromName
#    Path: Will get a dataset from path -> DatasetFromPath
#    tuple: Will merge datasets specified in the tuple -> DatasetMerger
#    list: Will define a list of transformations -> Transforms

split_ratio = {"train": 0.8, "val": 0.1, "test": 0.1}
implicit_dataset_specification = (
    ["mnist", SelectSamples(n_samples=20), Shuffle(seed=123)],
    [Path(".data/datasets/mnist"), SelectSamples(n_samples=30), SplitsByRatios(split_ratios=split_ratio)],
    ("mnist", "mnist"),
)

# Test the conversion function
explicit_dataset_specification = convert_to_explicit_specification(implicit_dataset_specification)
rprint("Converted explicit specification:")
rprint(explicit_dataset_specification)

assert explicit_dataset_specification == dataset_specification_again
