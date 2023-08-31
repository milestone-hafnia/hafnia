# Milestone Data Insight Platform

A command-line tool and Python package to download and load datasets from AWS S3.

## Installation

Clone the repository and navigate into it:

```bash
git clone https://github.com/Data-insight-Platform/mdi-cli.git
cd mdi-cli
```

Then, install the package using pip:

```bash
pip install -e .
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

This command downloads a dataset from S3, if it's not already downloaded, and loads it into a PyTorch DataLoader. It has a `--force` option to force a re-download of the data.

```bash
mdi load_dataset --name cats_and_dogs
```

To force a re-download of the data, use the --force option:

```bash
mdi load_dataset --name cats_and_dogs --force
```

### Python Package

You can also use `mdi` as a Python package.

Here's an example of how to download a dataset and load it into a PyTorch DataLoader:

```python
import mdi

# Download the dataset and load it into a PyTorch DataLoader
dataset = mdi.load_dataset("cats_and_dogs")

# Use the dataset for training a model
for images, labels in dataset["train"]:
    # training loop
    pass
```

## License

### MIT

Please make sure to replace "username" with your actual GitHub username in the repository link. Also, you might want to add more information such as how to contribute to your project, code of conduct, etc., depending on the nature of your project.
