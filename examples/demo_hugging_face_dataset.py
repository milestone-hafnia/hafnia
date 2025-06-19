import collections

import datasets
import rich
from datasets import ClassLabel

# Load dataset splits
dataset_splits = datasets.load_from_disk(".data/datasets/midwest-vehicle-detection/sample")

# Get the train split
dataset_train = dataset_splits["train"]

# Hugging face datasets have features - that describe all columns in the dataset - including class labels
rich.print(dataset_train.features)  # Print all features of the dataset

# Get hugging face 'ClassLabel' feature
class_labels: ClassLabel = dataset_train.features["objects"].feature["class_idx"]

# Class names
print("Class names:", class_labels.names)

# Mapping from class index to class name
class_index = 0
class_name_from_index = class_labels.int2str(class_index)
print(f"Class name for index '{class_index}' -> '{class_name_from_index}'")


# The reverse mapping from class name to class index
class_name = "Vehicle.Bicycle"
class_index_from_name = class_labels.str2int(class_name)
print(f"Class index for name '{class_name}' -> '{class_index}'")

# Flatten object names
object_names = []
for obj in dataset_train["objects"]:
    object_names.extend(obj["class_name"])

class_counts = collections.Counter(object_names)
rich.print(class_counts)
