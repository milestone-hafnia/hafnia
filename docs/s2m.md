# Script2Model: Converting Scripts to Models

Script2Model is a utility in the Hafnia Python Tools that helps you convert your Python scripts into deployable models on the Hafnia platform.

## Overview

Script2Model simplifies the process of converting experimental code into production-ready models by:

1. Analyzing your Python script
2. Extracting the necessary dependencies and functionality
3. Creating a standardized model structure
4. Packaging the model for deployment on the Hafnia platform

## Prerequisites

- An active Hafnia platform profile (configure with `hafnia configure`)
- Python scripts with well-defined inputs and outputs
- Required dependencies installed

## Basic Usage

To convert a Python script to a model:

```bash
hafnia script2model convert <script_path> [--output <output_dir>] [--name <model_name>]
```

### Parameters

- `<script_path>`: Path to your Python script
- `--output <output_dir>`: (Optional) Directory where the model will be saved (defaults to "./model")
- `--name <model_name>`: (Optional) Name for the model (defaults to script filename)

## Example

```bash
# Convert a training script to a model
hafnia script2model convert ./src/scripts/train.py --name my-classifier

# Deploy the model to Hafnia platform
hafnia script2model deploy ./model --experiment-id exp-123456
```

## Working with Model Configuration

Script2Model automatically detects function signatures and generates a configuration template. You can customize this configuration:

```bash
# Generate a configuration template
hafnia script2model config-template ./src/scripts/train.py

# Convert with a custom configuration
hafnia script2model convert ./src/scripts/train.py --config ./config.json
```

## Best Practices

1. **Input/Output Documentation**: Ensure your script has clear docstrings documenting input and output parameters
2. **Function Organization**: Use well-structured functions with clear purposes
3. **Error Handling**: Implement proper error handling in your script
4. **Dependencies**: List all dependencies in requirements.txt or environment.yml
5. **Testing**: Test your script locally before conversion

## Logging with HafniaLogger

Script2Model integrates with HafniaLogger to track model training and evaluation metrics:

```python
from hafnia.experiment import HafniaLogger

# Initialize logger
logger = HafniaLogger(Path("./logs"), update_interval=5)

# Log metrics during training
logger.log_metric("accuracy", value=0.95, step=100)
logger.log_metric("loss", value=0.05, step=100)

# Log configuration
logger.log_configuration({
    "learning_rate": 0.001,
    "batch_size": 32,
    "model_type": "resnet50"
})
```
