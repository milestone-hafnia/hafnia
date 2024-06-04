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

The `mdi` CLI has two main commands: `login` and `load-dataset`.

#### Login

This is interactive command that prompts the user for an API key and stores it for future use:

```bash
mdi login
```

**Note:** for non interactive login and multiple login profiles support check: ```mdi profile --help```

#### Load Dataset

This command downloads a dataset from S3, if it's not already downloaded, and loads it as Huggingface Dataset. It has a `--force` option to force a re-download of the data.

```bash
mdi load-dataset --name mnist
```

To force a re-download of the data, use the --force option:

```bash
mdi load-dataset --name mnist --force
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

Initialize and activate virtual env:
```bash
python -m venv .venv
source .venv/bin/activate
```

Then, install the package using poetry:

```bash
poetry install
```

To run the module use:
```bash
python -m mdi
```

To install the module with `pipx` use:
```bash
pipx install .
```


## Developments notes

For easier development we have a devcontainers configuration located in `.devcontainer` directory. For better portability of the project, we are targeting an older version of Python (specified in the devcontainer config).

When contributing, follow the best practices for the development of CLI tools. Some resources:
- https://clig.dev
- https://medium.com/@jdxcode/12-factor-cli-apps-dd3c227a0e46

