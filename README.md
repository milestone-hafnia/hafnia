# Hafnia

The `hafnia` python package is a collection of tools to create and run model training recipes on
the [Hafnia Platform](https://hafnia.milestonesys.com/). 

The package includes the following interfaces: 

- `cli`: A Command Line Interface (CLI) to 1) configure/connect to Hafnia's [Training-aaS](https://hafnia.readme.io/docs/training-as-a-service) and 2) create and 
launch recipe scripts.
- `hafnia`: A python package with helper functions to load and interact with sample datasets and an experiment
 tracker (`HafniaLogger`). 


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
    hafnia data download mnist --force
    ```

## Getting started: Loading datasets samples
With Hafnia configured on your local machine, it is now possible to download 
and explore the dataset sample with a python script:

```python
from hafnia.data import load_dataset

dataset_splits = load_dataset("mnist")
```

### Dataset Format
The returned sample dataset is a [hugging face dataset](https://huggingface.co/docs/datasets/index) 
and contains train, validation and test splits. 

```python
print(dataset_splits)

# Output:
>>> DatasetDict({
    train: Dataset({
        features: ['image_id', 'image', 'height', 'width', 'objects', 'Weather', 'Surface Conditions'],
        num_rows: 172
    })
    validation: Dataset({
        features: ['image_id', 'image', 'height', 'width', 'objects', 'Weather', 'Surface Conditions'],
        num_rows: 21
    })
    test: Dataset({
        features: ['image_id', 'image', 'height', 'width', 'objects', 'Weather', 'Surface Conditions'],
        num_rows: 21
    })
})

```

A Hugging Face dataset is a dictionary with splits, where each split is a `Dataset` object.
Each `Dataset` is structured as a table with a set of columns (also called features) and a row for each sample.

The features of the dataset can be viewed with the `features` attribute of the dataset split.
```python
# View features of the train split
pprint.pprint(dataset["train"].features)
{'Surface Conditions': ClassLabel(names=['Dry', 'Wet'], id=None),
 'Weather': ClassLabel(names=['Clear', 'Foggy'], id=None),
 'height': Value(dtype='int64', id=None),
 'image': Image(mode=None, decode=True, id=None),
 'image_id': Value(dtype='int64', id=None),
 'objects': Sequence(feature={'bbox': Sequence(feature=Value(dtype='int64',
                                                             id=None),
                                               length=-1,
                                               id=None),
                              'class_idx': ClassLabel(names=['Vehicle.Bicycle',
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
                                                             'Vehicle.Heavy_Duty_Vehicle'],
                                                      id=None),
                              'class_name': Value(dtype='string', id=None),
                              'id': Value(dtype='string', id=None)},
                     length=-1,
                     id=None),
 'width': Value(dtype='int64', id=None)}
```
In above representation, the `features` attribute shows the columns of the dataset split.

View the first sample in the training set:
```python
# Print sample from the training set
pprint.pprint(dataset["train"][0])

{'image': <PIL.PngImagePlugin.PngImageFile image mode=RGB size=1920x1080 at 0x79D6292C5ED0>,
 'image_id': 4920,
 'height': 1080,
 'Weather': 0,
 'Surface Conditions': 0,
 'objects': {'bbox': [[441, 180, 121, 126],
                      [549, 151, 131, 103],
                      [1845, 722, 68, 130],
                      [1810, 571, 110, 149]],
             'class_idx': [7, 7, 2, 2],
             'class_name': ['Vehicle.Pickup_Truck',
                            'Vehicle.Pickup_Truck',
                            'Vehicle.Car',
                            'Vehicle.Car'],
             'id': ['HW6WiLAJ', 'T/ccFpRi', 'CS0O8B6W', 'DKrJGzjp']},
 'width': 1920}

```

For hafnia based datasets, we want to standardized how a dataset and dataset tasks are represented.
We have defined a set of features that are common across all datasets in the Hafnia data library.

- `image`: The image itself, stored as a PIL image
- `height`: The height of the image in pixels
- `width`: The width of the image in pixels
- `[IMAGE_CLASSIFICATION_TASK]`: [Optional] Image classifications tasks is a top-level `ClassLabel` feature. The feature can be used for mapping image class indices to class names. In above example we have two classification tasks:
  - `Weather`: Classifies the weather conditions in the image, with possible values `Clear` and `Foggy`
  - `Surface Conditions`: Classifies the surface conditions in the image, with possible values `Dry` and `Wet`
- `objects`: A dictionary containing information about objects in the image, including:
  - `bbox`: Bounding boxes for each object, represented as a list of lists with coordinates `[xmin, ymin, bbox_width, bbox_height]`. It is the top-left corner `(xmin, ymin)` of the bounding box and its width and height `(bbox_width, bbox_height)` in pixels.
  - `class_idx`: Class indices for each detected object. This is a
  `ClassLabel` feature that maps to the `class_name` feature.
  - `class_name`: Class names for each detected object
  - `id`: Unique identifiers for each detected object

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
uv sync
```

 Run tests:
```bash
uv run pytest tests
```
