# The Hafnia Dataset Format

`HafniaDataset` is the in-memory representation of a computer-vision dataset
used throughout the Hafnia SDK. It is the same object whether you are working
with a small **sample dataset** locally or with a **full dataset** under
Training-aaS — a training script written against `HafniaDataset` does not
change between the two environments.

## What a dataset contains

A `HafniaDataset` is a pair of two fields:

- **`dataset.info`** — a `DatasetInfo` describing the dataset and its tasks.
  Each `TaskInfo` declares a primitive (`Bbox`, `Bitmask`, `Polygon`,
  `Classification`, ...) and the list of `ClassInfo` entries that define the
  label space for that task.
- **`dataset.samples`** — a [Polars](https://pola.rs/) `DataFrame` where each
  row is one image (or video frame). Primitive columns (`classifications`,
  `bboxes`, `bitmasks`, `polygons`) are stored as `List[Struct]` and mirror
  the fields of the corresponding `Sample` Pydantic model.

A row in `dataset.samples` unpacks into a `Sample` object that carries the
image path, dimensions, split, tags, optional camera/position/orientation
metadata, and the per-sample annotation primitives.

## Loading and writing

```python
from hafnia.dataset.hafnia_dataset import HafniaDataset

# Load by name — sample dataset locally, full dataset under Training-aaS
dataset = HafniaDataset.from_name("midwest-vehicle-detection", version="1.0.0")

# Or load from disk
dataset = HafniaDataset.from_path(".data/datasets/midwest-vehicle-detection")

# Persist to disk - images are stored as real copies
dataset.write(".data/tmp/hafnia_dataset")

# ... or as symbolic links to avoid duplicating a large dataset on disk. Note that the
# written dataset breaks if the original files are moved or deleted. Also available for
# 'to_yolo_format', 'to_coco_format' and 'to_image_classification_folder'.
dataset.write(".data/tmp/hafnia_dataset", storage_mode="symlink")  # Or FileStorageMode.SYMLINK
```

`from_name` downloads the dataset to `.data/datasets/` on first call and
caches it for subsequent calls.

### Loading from sample files

`from_path` reads a dataset folder written by `write`, with all annotations in a
single annotations file. `from_paths` instead loads a dataset from a list of sample
files located anywhere — e.g. one annotation or prediction file per image or video,
which can be produced independently and in parallel:

```
my_annotations/
├── dataset_info.json        # DatasetInfo (tasks, class names, ...)
├── image0.json              # A single Sample ({"file_path": "image0.jpg", ...})
├── image0.jpg
├── image1.jsonl             # One Sample per line, e.g. multiple predictions for the same image
├── image1.jpg
├── video0.jsonl             # One Sample per frame of a video
└── video0.mp4
```

```python
from pathlib import Path

from hafnia.dataset.hafnia_dataset import HafniaDataset

folder = Path("my_annotations")
dataset = HafniaDataset.from_paths(
    sample_paths=[folder / "image0.json", folder / "image1.jsonl", folder / "video0.jsonl"],
    dataset_info=folder / "dataset_info.json",  # Or a 'DatasetInfo' object
    media_paths=[None, None, folder / "video0.mp4"],  # Optional: overrules 'file_path' of all samples in a file
    reindex_class_idx=True,  # Optional: derive 'class_idx' from 'class_name' and 'dataset_info.tasks'
)
```

- A `.json` file holds a single `Sample` object or a list of samples; a `.jsonl` file holds one sample per line.
- A relative `file_path` in a sample is resolved against the folder of its sample file.
- The annotations are validated against `dataset_info`. As the sample files may be a subset of a dataset,
  tasks in `dataset_info` without annotations are allowed.
- Single sample files can be written with `Sample.to_file` / `Sample.samples_to_file` and read without
  a `HafniaDataset` with `Sample.samples_from_file` / `Sample.samples_from_files`.

Writing a full dataset as separate sample files is not supported — use `write` to store a full dataset.

## Iterating samples

```python
from hafnia.dataset.hafnia_dataset_types import Sample

for sample_dict in dataset:
    sample = Sample(**sample_dict)   # validate + IDE/mypy support
    image = sample.read_image()      # np.ndarray
    for bbox in sample.bboxes or []:
        print(bbox.class_name, bbox.top_left_x, bbox.top_left_y)
    break
```

`Sample` also exposes `draw_annotations()` for quick visualization.

## Common operations

`HafniaDataset` supports chainable operations that return a new dataset:

```python
dataset_train = dataset.create_split_dataset("train")
small        = dataset.select_samples(n_samples=10, seed=42)
shuffled     = dataset.shuffle(seed=42)
splits       = dataset.splits_by_ratios({"train": 0.8, "val": 0.1, "test": 0.1})
only_ones    = dataset.select_samples_by_class_name(name="1 - one", primitive="Classification")
remapped     = dataset.class_mapper(class_mapping={"car": "Vehicle", "person": "Person"})
merged       = HafniaDataset.from_merge(dataset0=a, dataset1=b)
```

For statistics, see `dataset.print_stats()`, `dataset.print_class_distribution()`,
`dataset.calculate_class_counts()` and friends. For anything custom, work
directly on `dataset.samples` with native Polars expressions.

## Annotation primitives

All annotations inherit from a common `Primitive` base and live in
[src/hafnia/dataset/primitives/](../src/hafnia/dataset/primitives/):

| Primitive        | Column            | Use                                    |
| ---------------- | ----------------- | -------------------------------------- |
| `Classification` | `classifications` | Image- or sample-level labels          |
| `Bbox`           | `bboxes`          | 2D bounding boxes (normalized coords)  |
| `Polygon`        | `polygons`        | Polygonal segmentations                |
| `Bitmask`        | `bitmasks`        | Pixel masks                            |

Each primitive carries `class_name`, `class_idx`, and a `ground_truth` flag.
Set `ground_truth=False` and add a `confidence` value to store predictions
alongside ground truth in the same dataset.

## Importers and exporters

Built-in converters under
[src/hafnia/dataset/format_conversions/](../src/hafnia/dataset/format_conversions/)
allow round-tripping with common formats:

```python
dataset.to_yolo_format(path_output=".data/tmp/yolo_dataset")
HafniaDataset.from_yolo_format(".data/tmp/yolo_dataset")

dataset.to_coco_format(path_output=".data/tmp/coco_dataset")
HafniaDataset.from_coco_format(".data/tmp/coco_dataset")
```

To bring data that is not yet on the platform into the Hafnia format, build
samples directly from `Sample`, `Bbox`, `DatasetInfo`, etc. See
[examples/example_custom_dataset.py](../examples/example_custom_dataset.py).

## See also

- [examples/example_hafnia_dataset.py](../examples/example_hafnia_dataset.py) — runnable walkthrough.
- [docs/dataset_recipe.md](dataset_recipe.md) — composing reproducible datasets from recipes.
