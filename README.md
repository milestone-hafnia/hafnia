# Milestone Data Insight Platform

# Developer environement setup.

1. Package manager is [uv](https://github.com/astral-sh/uv). Install with `pip install uv`.
2. Install dependencies with `uv install`.
3. Code formatting and linting is done with `pre-commit`. Install it first and run `pre-commit run --all-files` to format and lint the code or it will be done before git commit.


### MDI CLI Tool Documentation 
## Command Overview

The MDI CLI tool is organized into two main command groups: `sys` for system management and `runc` for experiment management.

### System Management Commands (`sys`)

| Command | Description | Arguments | Options | Example |
|---------|-------------|-----------|----------|---------|
| `sys configure` | Configure MDI CLI settings | None | None | `mdi sys configure`<br>Prompts for MDI API Key (hidden input) |
| `sys profile` | Display current configuration | None | None | `mdi sys profile`<br>Shows masked API key: "abcd...wxyz" |
| `sys clear` | Remove stored configuration | None | None | `mdi sys clear`<br>Removes all stored settings |

### [PRIVATE] Experiment Management Commands (`runc`) 

| Command | Description | Arguments | Options | Example |
|---------|-------------|-----------|----------|---------|
| `runc status` | Check experiment status | None | `--verbose`: Verbosity level (default: 1) | `mdi runc status --verbose 2` |
| `runc mount` | Mount user source | `SOURCE`: Path to source (required) | None | `mdi runc mount ./my_experiment` |
| `runc prepare` | Prepare experiment environment | `SOURCE`: Source path (required)<br>`EXEC_CMD`: Execution command (required) | None | `mdi runc prepare ./my_experiment "train"` |
| `runc launch` | Launch experiment job | `TASK`: Task identifier (required) | None | `mdi runc launch training_task_1` |


# README LEGACY
---------

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

This is an interactive command that prompts the user for an API key and stores it for the future use:

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

## Configuration with environment variables

There are a few optional environment variables that can be used for configuration.

| Environment variable | Description                                                     |
| -------------------- | --------------------------------------------------------------- |
| MDI_API_KEY          | API key to use instead of reading from the config profile       |
| MDI_API_URL          | API url to use instead of reading from the config profile       |
| MDI_CONFIG_FILE      | The location of the configuration file (`config.ini`)           |
| MDI_CACHE_DIR        | The location of the directory where datasets will be downloaded |


## Installation Development Version

Clone the repository and navigate into it:

```bash
git clone https://github.com/Data-insight-Platform/mdi-cli.git
cd mdi-cli
```

Then, install the package using poetry:

```bash
poetry install
```

Activate the virtual env (created by poetry):
```bash
source .venv/bin/activate
```

To run the module use:
```bash
python -m mdi
```

To install the module from the local directory with `pipx` use:
```bash
pipx install .
```


## Developments notes

For easier development we have a devcontainers configuration located in `.devcontainer` directory. For better portability of the project, we are targeting an older version of Python (specified in the devcontainer config).

When contributing, follow the best practices for the development of CLI tools. Some resources:
- https://clig.dev
- https://medium.com/@jdxcode/12-factor-cli-apps-dd3c227a0e46

