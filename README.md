# Milestone Data Insight Platform

A command-line tool and Python package to download and load datasets (Huggingface format) from the MDI platform.

## Installation

With pip:

```bash
pip install git+ssh://git@github.com/Data-insight-Platform/mdi-cli.git@main
```

With poetry, add it to the `pyproject.toml` file of your project:

```
[tool.poetry.dependencies]
...
mdi = { git = "https://github.com/Data-insight-Platform/mdi-cli.git", branch = "main" }
...
```

## Usage
### Command-Line Interface

The `mdi` CLI has two commands: `login` and `load_dataset`.

#### Login

This command prompts the user for an API key and stores it for future use:

```bash
mdi login
```

#### Load Dataset

This command downloads a dataset from S3, if it's not already downloaded, and loads it as Huggingface Dataset. It has a `--force` option to force a re-download of the data.

```bash
mdi load_dataset --name mnist
```

To force a re-download of the data, use the --force option:

```bash
mdi load_dataset --name mnist --force
```

### Python Package

You can also use `mdi` as a Python package.

Here's an example of how to download a dataset and load it as Huggingface Dataset:

```python
import mdi

# Download the dataset and load it as Huggingface Dataset
dataset = mdi.load_dataset("mnist")

# Use the dataset for training a model
for images, labels in dataset["train"]:
    # training loop
    pass
```

## Installation Development Version

Clone the repository and navigate into it:

```bash
git clone https://github.com/Data-insight-Platform/mdi-cli.git
cd mdi-cli
```

Then, install the package using pip:

```bash
pip install -e .
```
