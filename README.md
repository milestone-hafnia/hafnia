# Hafnia

The `hafnia` python package is a collection of tools to create and run model training recipes on
the [Hafnia Platform](https://hafnia.milestonesys.com/). 

The package includes the following interfaces: 

- `cli`: A Command Line Interface (CLI) to 1) configure/connect to Hafnia's [Training-aaS](https://hafnia.readme.io/docs/training-as-a-service) and 2) create and 
launch recipe scripts.
- `hafnia`: A python package including `HafniaDataset` to manage datasets and `HafniaLogger` to do 
experiment tracking.


## The Concept: Training as a Service (Training-aaS)
`Training-aaS` is the concept of training models on the Hafnia platform on large 
and *hidden* datasets. Hidden datasets refers to datasets that can be used for 
training, but are not available for download or direct access. 

This is a key feature of the Hafnia platform, as a hidden dataset ensures data 
privacy, and allow models to be trained compliantly and ethically by third parties (you).

The `script2model` approach is a Training-aaS concept, where you package your custom training 
script as a *training recipe* and use the recipe to train models on the hidden datasets.

To support local development of a training recipe, we have introduced a **sample dataset** 
for each dataset available in the Hafnia [data library](https://hafnia.milestonesys.com/training-aas/datasets). The sample dataset is a small 
and anonymized subset of the full dataset and available for download. 

With the sample dataset, you can seamlessly switch between local development and Training-aaS. 
Locally, you can create, validate and debug your training recipe. The recipe is then 
launched with Training-aaS, where the recipe runs on the full dataset and can be scaled to run on
multiple GPUs and instances if needed. 

## Getting started: Configuration
To get started with Hafnia: 

1. Install `hafnia` with your favorite python package manager. With pip do this:

    `pip install hafnia`
1. Sign in to the [Hafnia Platform](https://hafnia.milestonesys.com/). 
1. Create an API KEY for Training aaS. For more instructions, follow this 
[guide](https://hafnia.readme.io/docs/create-an-api-key). 
Copy the key and save it for later use.
1. From terminal, configure your machine to access Hafnia: 

    ```
    # Start configuration with
    hafnia configure

    # You are then prompted: 
    Profile Name [default]:   # Press [Enter] or select an optional name
    Hafnia API Key:  # Pass your HAFNIA API key
    Hafnia Platform URL [https://api.mdi.milestonesys.com]:  # Press [Enter]
    ```
1. Download `mnist` from terminal to verify that your configuration is working.  

    ```bash
    hafnia dataset download mnist --force
    ```

## Getting started: Loading datasets samples
With Hafnia configured on your local machine, it is now possible to download 
and explore the dataset sample with a python script:

```python
from hafnia.data import load_dataset, get_dataset_path
from hafnia.dataset.hafnia_dataset import HafniaDataset

# To download the sample dataset use:
path_dataset = get_dataset_path("midwest-vehicle-detection")
```

This will download the dataset sample `midwest-vehicle-detection` to the local `.data/datasets/` folder
in a human readable format. 

Images are stored in the `data` folder, general dataset information is stored in `dataset_info.json`  
and annotations are stored as both `annotations.jsonl`(jsonl) and `annotations.parquet`.

```bash
$ cd .data/datasets/
$ tree midwest-vehicle-detection
midwest-vehicle-detection
└── sample
    ├── annotations.jsonl
    ├── annotations.parquet
    ├── data
    │   ├── video_0026a86b-2f43-49f2-a17c-59244d10a585_1fps_mp4_frame_00000.png
    │   ....
    │   ├── video_ff17d777-e783-44e2-9bff-a4adac73de4b_1fps_mp4_frame_00000.png
    │   └── video_ff17d777-e783-44e2-9bff-a4adac73de4b_1fps_mp4_frame_00100.png
    └── dataset_info.json

3 directories, 217 files
```

You can interact with data as you want, but we also provide `HafniaDataset`
for loading/saving, managing and interacting with the dataset.

We recommend to visit and potentially execute the example script [examples/example_hafnia_dataset.py](examples/example_hafnia_dataset.py) 
to see how to use the `HafniaDataset` class and its methods.

Below is a short introduction to the `HafniaDataset` class.

```python
from hafnia.dataset.hafnia_dataset import HafniaDataset, Sample

# Load dataset
dataset = HafniaDataset.read_from_path(path_dataset)

# Alternatively, you can use the 'load_dataset' function to download and load dataset in one go.
# dataset = load_dataset("midwest-vehicle-detection")

# Print dataset information
dataset.print_stats()

# Create a dataset split for training
dataset_train = dataset.create_split_dataset("train")
```

The `HafniaDataset` object provides a convenient way to interact with the dataset, including methods for 
creating splits, accessing samples, printing statistics, saving to and loading from disk.

In essence, the `HafniaDataset` class contains `dataset.info` with dataset information 
and `dataset.table` with annotations as a polars DataFrame

```python
# Annotations are stored in a polars DataFrame
print(dataset.table.head(2))
shape: (2, 14)
┌──────────┬────────────────────────────────┬────────┬───────┬───┬───────────────────────────────┬──────────┬──────────┬───────────────────────────────┐
│ image_id ┆ file_name                      ┆ height ┆ width ┆ … ┆ objects                       ┆ bitmasks ┆ polygons ┆ meta                          │
│ ---      ┆ ---                            ┆ ---    ┆ ---   ┆   ┆ ---                           ┆ ---      ┆ ---      ┆ ---                           │
│ str      ┆ str                            ┆ i64    ┆ i64   ┆   ┆ list[struct[12]]              ┆ null     ┆ null     ┆ struct[5]                     │
╞══════════╪════════════════════════════════╪════════╪═══════╪═══╪═══════════════════════════════╪══════════╪══════════╪═══════════════════════════════╡
│ 7800     ┆ /home/ubuntu/code/hafnia/.data ┆ 1080   ┆ 1920  ┆ … ┆ [{0.0492,0.0357,0.2083,0.23," ┆ null     ┆ null     ┆ {120.0,1.0,"2024-07-10T18:30: │
│          ┆ …                              ┆        ┆       ┆   ┆ V…                            ┆          ┆          ┆ 0…                            │
│ 7900     ┆ /home/ubuntu/code/hafnia/.data ┆ 1080   ┆ 1920  ┆ … ┆ [{0.146382,0.078704,0.42963,0 ┆ null     ┆ null     ┆ {120.0,1.0,"2024-07-10T18:30: │
│          ┆ …                              ┆        ┆       ┆   ┆ .…                            ┆          ┆          ┆ 0…                            │
└──────────┴────────────────────────────────┴────────┴───────┴───┴───────────────────────────────┴──────────┴──────────┴───────────────────────────────┘
```

```python
# General dataset information is stored in `dataset.info`
rich.print(dataset.info)
DatasetInfo(
    dataset_name='midwest-vehicle-detection',
    version='1.0.0',
    tasks=[
        TaskInfo(
            primitive=<class 'hafnia.dataset.primitives.Bbox'>,
            class_names=[
                'Person',
                'Vehicle.Bicycle',
                'Vehicle.Motorcycle',
                'Vehicle.Car',
                'Vehicle.Van',
                'Vehicle.RV',
                'Vehicle.Single_Truck',
                'Vehicle.Combo_Truck',
                'Vehicle.Pickup_Truck',
                'Vehicle.Trailer',
                'Vehicle.Emergency_Vehicle',
                'Vehicle.Bus',
                'Vehicle.Heavy_Duty_Vehicle'
            ],
            name='bboxes'
        ),
        TaskInfo(primitive=<class 'hafnia.dataset.primitives.Classification'>, class_names=['Clear', 'Foggy'], name='Weather'),
        TaskInfo(primitive=<class 'hafnia.dataset.primitives.Classification'>, class_names=['Dry', 'Wet'], name='Surface Conditions')
    ],
    meta={
        'n_videos': 109,
        'n_cameras': 20,
        'duration': 13080.0,
        'duration_average': 120.0,
        ...
    }
)
```

You can iterate and access samples in the dataset using the `HafniaDataset` object.
Each sample contain image and annotations information. 

```python
from hafnia.dataset.hafnia_dataset import HafniaDataset, Sample
# Access the first sample in the dataset either by index or by iterating over the dataset
sample_dict = dataset[0]

for sample_dict in dataset:
    sample = Sample(**sample_dict)
    print(sample.sample_id, sample.objects)
    break
```
Not that it is possible to create a `Sample` object from the sample dictionary.
This is useful for accessing the image and annotations in a structured way.

```python
# By unpacking the sample dictionary, you can create a `Sample` object.
sample = Sample(**sample_dict)

# Use the `Sample` object to easily read image and draw annotations
image = sample.read_image()
image_annotations = sample.draw_annotations()
```

Note that the `Sample` object contains all information about the sample, including image and metadata.
It also contain annotations as primitive types such as `Bbox`, `Classification`. 

```python
rich.print(sample)
Sample(
    image_id='7800',
    file_name='data/video_0026a86b-2f43-49f2-a17c-59244d10a585_1fps_mp4_frame_0
0000.png',
    height=1080,
    width=1920,
    split='test',
    is_sample=True,
    frame_number=None,
    video_name=None,
    remote_path=None,
    classifications=[
        Classification(
            class_name='Clear',
            class_idx=0,
            object_id=None,
            confidence=None,
            ground_truth=True,
            task_name='Weather',
            meta=None
        ),
        Classification(
            class_name='Day',
            class_idx=3,
            object_id=None,
            confidence=None,
            ground_truth=True,
            task_name='Time of Day',
            meta=None
        ),
        ...
    ],
    objects=[
        Bbox(
            height=0.0492,
            width=0.0357,
            top_left_x=0.2083,
            top_left_y=0.23,
            class_name='Vehicle.Car',
            class_idx=3,
            object_id='cXT4NRVu',
            confidence=None,
            ground_truth=True,
            task_name='bboxes',
            meta=None
        ),
        Bbox(
            height=0.0457,
            width=0.0408,
            top_left_x=0.2521,
            top_left_y=0.2153,
            class_name='Vehicle.Car',
            class_idx=3,
            object_id='MelbIIDU',
            confidence=None,
            ground_truth=True,
            task_name='bboxes',
            meta=None
        ),
        ...
    ],
    bitmasks=None,  # Optional a list of Bitmask objects List[Bitmask]
    polygons=None,  # Optional a list of Polygon objects List[Polygon]
    meta={
        'video.data_duration': 120.0,
        'video.data_fps': 1.0,
        ...
    }
)
```

To learn more, view and potentially execute the example script [examples/example_hafnia_dataset.py](examples/example_hafnia_dataset.py). 

### Dataset Locally vs. Training-aaS
An important feature of `load_dataset` is that it will return the full dataset 
when loaded with Training-aaS on the Hafnia platform. 

This enables seamlessly switching between running/validating a training script 
locally (on the sample dataset) and running full model trainings with Training-aaS (on the full dataset). 
without changing code or configurations for the training script.

Available datasets with corresponding sample datasets can be found in [data library](https://hafnia.milestonesys.com/training-aas/datasets) including metadata and description for each dataset. 


## Getting started: Experiment Tracking with HafniaLogger
The `HafniaLogger` is an important part of the recipe script and enables you to track, log and
reproduce your experiments.

When integrated into your training script, the `HafniaLogger` is responsible for collecting:

- **Trained Model**: The model trained during the experiment
- **Model Checkpoints**: Intermediate model states saved during training
- **Experiment Configurations**: Hyperparameters and other settings used in your experiment
- **Training/Evaluation Metrics**: Performance data such as loss values, accuracy, and custom metrics

### Basic Implementation Example

Here's how to integrate the `HafniaLogger` into your training script:

```python
from hafnia.experiment import HafniaLogger

batch_size = 128
learning_rate = 0.001

# Initialize Hafnia logger
logger = HafniaLogger()

# Log experiment parameters
logger.log_configuration({"batch_size": 128, "learning_rate": 0.001})

# Store checkpoints in this path
ckpt_dir = logger.path_model_checkpoints()

# Store the trained model in this path
model_dir = logger.path_model()

# Log scalar and metric values during training and validation
logger.log_scalar("train/loss", value=0.1, step=100)
logger.log_metric("train/accuracy", value=0.98, step=100)

logger.log_scalar("validation/loss", value=0.1, step=100)
logger.log_metric("validation/accuracy", value=0.95, step=100)
```

Similar to `load_dataset`, the tracker behaves differently when running locally or in the cloud. 
Locally, experiment data is stored in a local folder `.data/experiments/{DATE_TIME}`. 

In the cloud, the experiment data will be available in the Hafnia platform under 
[experiments](https://hafnia.milestonesys.com/training-aas/experiments). 

## Example: Torch Dataloader
Commonly for `torch`-based training scripts, a dataset is used in combination 
with a dataloader that performs data augmentations and batching of the dataset as torch tensors.

To support this, we have provided a torch dataloader example script
[example_torchvision_dataloader.py](./examples/example_torchvision_dataloader.py). 

The script demonstrates how to load a dataset sample, apply data augmentations using
`torchvision.transforms.v2`, and visualize the dataset with `torch_helpers.draw_image_and_targets`.

Note also how `torch_helpers.TorchVisionCollateFn` is used in combination with the `DataLoader` from 
`torch.utils.data` to handle the dataset's collate function.

The dataloader and visualization function supports computer vision tasks 
and datasets available in the data library. 

```python
# Load Hugging Face dataset
dataset_splits = load_dataset("midwest-vehicle-detection")

# Define transforms
train_transforms = v2.Compose(
    [
        v2.RandomResizedCrop(size=(224, 224), antialias=True),
        v2.RandomHorizontalFlip(p=0.5),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)
test_transforms = v2.Compose(
    [
        v2.Resize(size=(224, 224), antialias=True),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

keep_metadata = True
train_dataset = torch_helpers.TorchvisionDataset(
    dataset_splits["train"], transforms=train_transforms, keep_metadata=keep_metadata
)
test_dataset = torch_helpers.TorchvisionDataset(
    dataset_splits["test"], transforms=test_transforms, keep_metadata=keep_metadata
)

# Visualize sample
image, targets = train_dataset[0]
visualize_image = torch_helpers.draw_image_and_targets(image=image, targets=targets)
pil_image = torchvision.transforms.functional.to_pil_image(visualize_image)
pil_image.save("visualized_labels.png")

# Create DataLoaders - using TorchVisionCollateFn
collate_fn = torch_helpers.TorchVisionCollateFn(
    skip_stacking=["objects.bbox", "objects.class_idx"]
)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)
```


## Example: Training-aaS
By combining logging and dataset loading, we can now construct our model training recipe. 

To demonstrate this, we have provided a recipe project that serves as a template for creating and structuring training recipes
[recipe-classification](https://github.com/milestone-hafnia/recipe-classification)

The project also contains additional information on how to structure your training recipe, use the `HafniaLogger`, the `load_dataset` function and different approach for launching 
the training recipe on the Hafnia platform.


## Create, Build and Run `recipe.zip` locally
In order to test recipe compatibility with Hafnia cloud use the following command to build and 
start the job locally.

```bash
    # Create 'recipe.zip' from source folder '.'
    hafnia recipe create .
    
    # Build the docker image locally from a 'recipe.zip' file
    hafnia runc build-local recipe.zip

    # Execute the docker image locally with a desired dataset
    hafnia runc launch-local --dataset mnist  "python scripts/train.py"
```

## Detailed Documentation
For more information, go to our [documentation page](https://hafnia.readme.io/docs/welcome-to-hafnia) 
or in below markdown pages. 

- [CLI](docs/cli.md) - Detailed guide for the Hafnia command-line interface
- [Release lifecycle](docs/release.md) - Details about package release lifecycle.

## Development
For development, we are using an uv based virtual python environment.

Install uv
```bash 
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Create virtual environment and install python dependencies

```bash
uv sync --dev
```

 Run tests:
```bash
uv run pytest tests
```
