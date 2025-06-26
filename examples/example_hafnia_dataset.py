from pathlib import Path
from typing import List

import numpy as np
from PIL import Image
from rich import print as rprint

from hafnia.data import get_dataset_path, load_dataset
from hafnia.dataset.dataset_names import SplitName
from hafnia.dataset.hafnia_dataset import DatasetInfo, HafniaDataset, Sample, TaskInfo
from hafnia.dataset.primitives.bbox import Bbox
from hafnia.dataset.primitives.bitmask import Bitmask
from hafnia.dataset.primitives.classification import Classification
from hafnia.dataset.primitives.polygon import Polygon

# First ensure that you have the Hafnia CLI installed and configured.
# You can install it via pip:
#   pip install hafnia
# And configure it with your Hafnia account:
#   hafnia configure

# Load dataset
path_dataset = get_dataset_path("midwest-vehicle-detection")
dataset = HafniaDataset.read_from_path(path_dataset)

# Alternatively, you can use the 'load_dataset' function
dataset = load_dataset("midwest-vehicle-detection")


# Dataset information is stored in 'dataset.info'
rprint(dataset.info)

# Annotations are stored in 'dataset.table' as a Polars DataFrame
dataset.samples.head(2)

# Print dataset information
dataset.print_stats()

# Create a dataset split for training
dataset_train = dataset.create_split_dataset("train")

# Checkout built-in transformations in 'dataset_transformation' or 'HafniaDataset'
dataset_val = dataset.create_split_dataset(SplitName.VAL)  # Use 'SplitName' to avoid magic strings

small_dataset = dataset.sample(n_samples=10, seed=42)  # Sample 10 samples from the dataset
shuffled_dataset = dataset.shuffle(seed=42)  # Shuffle the dataset

split_ratios = {SplitName.TRAIN: 0.8, SplitName.VAL: 0.1, SplitName.TEST: 0.1}
new_dataset_splits = dataset.split_by_ratios(split_ratios)

# Write dataset to disk
path_tmp = Path(".data/tmp")
path_dataset = path_tmp / "hafnia_dataset"
dataset.write(path_dataset)  # --> Check that data is human readable

# Load dataset from disk
dataset_again = HafniaDataset.read_from_path(path_dataset)

# Access the first sample in the training split - data is stored in a dictionary
sample_dict = dataset_train[0]

# Dataset can also be iterated to get sample data
for sample_dict in dataset_train:
    break

# Unpack dict into a Sample-object! Important for data validation, useability, IDE completion and mypy hints
sample: Sample = Sample(**sample_dict)

objects: List[Bbox] = sample.objects  # Use 'sample.objects' access bounding boxes as a list of Bbox objects
bitmasks: List[Bitmask] = sample.bitmasks  # Use 'sample.bitmasks' to access bitmasks as a list of Bitmask objects
polygons: List[Polygon] = sample.polygons  # Use 'sample.polygons' to access polygons as a list of Polygon objects
classifications: List[Classification] = sample.classifications  # As a list of Classification objects

# Read image using the sample object
image: np.ndarray = sample.read_image()

# Visualize sample and annotations
image_with_annotations = sample.draw_annotations()


path_tmp.mkdir(parents=True, exist_ok=True)
Image.fromarray(image_with_annotations).save(path_tmp / "sample_with_annotations.png")


# Do dataset transformations and statistics on the Polars DataFrame
n_objects = dataset.samples["objects"].list.len().sum()
n_objects = dataset.samples[Bbox.column_name()].list.len().sum()  # Use Bbox.column_name() to avoid magic variables
n_classifications = dataset.samples[Classification.column_name()].list.len().sum()

class_counts = dataset.samples[Classification.column_name()].explode().struct.field("class_name").value_counts()
class_counts = dataset.samples[Bbox.column_name()].explode().struct.field("class_name").value_counts()
rprint(dict(class_counts.iter_rows()))


## Bring-your-own-data: Create a new dataset from samples
fake_samples = []
for i_fake_sample in range(5):
    bboxes = [Bbox(top_left_x=10, top_left_y=20, width=100, height=200, class_name="car")]
    classifications = [Classification(class_name="vehicle", class_idx=0)]
    sample = Sample(
        file_name=f"path/to/image_{i_fake_sample:05}.jpg",
        height=480,
        width=640,
        split="train",
        is_sample=True,
        objects=bboxes,
        classifications=classifications,
    )
    fake_samples.append(sample)


fake_dataset_info = DatasetInfo(
    dataset_name="fake-dataset",
    version="0.0.1",
    tasks=[
        TaskInfo(primitive=Bbox, class_names=["car", "truck", "bus"]),
        TaskInfo(primitive=Classification, class_names=["vehicle", "pedestrian", "cyclist"]),
    ],
)
fake_dataset = HafniaDataset.from_samples_list(samples_list=fake_samples, info=fake_dataset_info)


## A hafnia dataset can also be used for storing predictions per sample set 'ground_truth=False' and add 'confidence'.
bboxes_predictions = [
    Bbox(top_left_x=10, top_left_y=20, width=100, height=200, class_name="car", ground_truth=False, confidence=0.9)
]

classifications_predictions = [Classification(class_name="vehicle", class_idx=0, ground_truth=False, confidence=0.95)]
