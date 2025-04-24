# Hafnia

The `hafnia` python package is a collection of tools to create and run model training recipes on
the [Hafnia Platform](https://hafnia.milestonesys.com/). 

The package includes the following interfaces: 

- `cli`: A Command Line Interface (CLI) to 1) configure/connect to Hafnia and 2) create and 
launch  [Training-aaS](https://hafnia.readme.io/docs/training-as-a-service) recipe scripts.
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

With the sample dataset, you can seamlessly switch between local and Hafnia training. 
Locally, you can create, validate and debug your training recipe. The recipe is then 
launched with Hafnia Training-aaS, where the recipe runs on the full dataset and can be scaled to run on
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
1. Download `mnist` from terminal to verify configuration is working.  

    ```bash
    hafnia data download mnist --force
    ```

## Getting started: Loading datasets samples
With Hafnia configured on your local machine, it is now possible to download 
and explore the dataset sample with a python script:

```python
from hafnia.data import load_dataset

dataset_splits = load_dataset("midwest-vehicle-detection")
print(dataset_splits)
print(dataset_splits["train"])
```
The returned sample dataset is a [hugging face dataset](https://huggingface.co/docs/datasets/index) 
and contains train, validation and test splits. 

An important feature of `load_dataset` is that it will return the full dataset 
when loaded on the Hafnia platform. 
This enables seamlessly switching between running/validating a training script 
locally (on the sample dataset) and running full model trainings in the cloud 
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

The script demonstrates how to make a dataloader with data augmentation (`torchvision.transforms.v2`)
and a helper function for visualizing image and labels. 

The dataloader and visualization function supports computer vision tasks 
and datasets available in the data library. 

## Example: Training-aaS
By combining logging and dataset loading, we can now construct our model training recipe. 

To demonstrate this, we have provided a recipe project that serves as a template for creating and structuring training recipes
[recipe-classification](https://github.com/Data-insight-Platform/recipe-classification)

The project also contains additional information on how to structure your training recipe, use the `HafniaLogger`, the `load_dataset` function and different approach for launching 
the training recipe on the Hafnia platform.

## Detailed Documentation
For more information, go to our [documentation page](https://hafnia.readme.io/docs/welcome-to-hafnia) 
or in below markdown pages. 

- [CLI](docs/cli.md) - Detailed guide for the Hafnia command-line interface
- [Script2Model Documentation](docs/s2m.md) - Detailed guide for script2model
- [Release lifecycle](docs/release.md) - Details about package release lifecycle.

## Development
For development, we are using an uv based virtual python environment.

Install uv

    curl -LsSf https://astral.sh/uv/install.sh | sh


Install python dependencies including developer (`--dev`) and optional dependencies (`--all-extras`).

    uv sync --all-extras --dev

 Run tests:

    uv run pytest tests
