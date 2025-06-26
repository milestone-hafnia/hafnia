A command-line interface tool for managing data science experiments and resources on the Project Hafnia.

## Features

- **Platform Configuration**: Easy setup and management of Hafnia platform settings

## Installation

## CLI Commands

### Core Commands

- `hafnia configure` - Configure Hafnia CLI settings
- `hafnia clear` - Remove stored configuration
- `hafnia profile` - Manage profiles (see subcommands below) 

### Profile Management

- `hafnia profile ls` - List all available profiles
- `hafnia profile use <profile_name>` - Switch to a different profile
- `hafnia profile rm <profile_name>` - Remove a specific profile
- `hafnia profile active` - Show detailed information about the active profile

### Dataset Management

- `hafnia dataset download mnist --force` - Download sample datasets

### Experiment Management

- `hafnia runc launch <task>` - Launch a job within the image
- `hafnia runc build <recipe_url> --st [state_file] --repo [registry/repo]` - Build docker image with a given recipe
- `hafnia runc build-local <recipe> --st [state_file] --repo [registry/repo | localhost]` - Build recipe from local path as image with prefix - localhost

## Configuration

he CLI tool supports multiple configuration profiles:

1. Run `hafnia configure`
2. Enter a profile name (defaults to "default")
3. Enter your Hafnia API Key when prompted
4. Provide the Hafnia Platform URL (defaults to "https://api.mdi.milestonesys.com")
5. The organization ID will be retrieved automatically
6. Verify your configuration with `hafnia profile active`

## Example Usage

```bash
# Configure the CLI with a new profile
hafnia configure

# List all available profiles
hafnia profile ls

# Switch to a different profile
hafnia profile use production

# View active profile details
hafnia profile active

# Remove a profile
hafnia profile rm old-profile

# Clear all configuration
hafnia clear

# Download a dataset sample
hafnia dataset download mnist --force

# Build a Docker image from a recipe
hafnia runc build https://api.mdi.milestonesys.com/api/v1/recipes/my-recipe

# Build a Docker image from a local recipe
hafnia runc build-local ./my-recipe

# Launch a task within the image
hafnia runc launch train
```

## Environment Variables

The CLI tool uses configuration stored in your local environment. You can view the current settings using:

```bash
hafnia profile active
```

Available environment variables:

- `MDI_CONFIG_PATH` - Custom path to the configuration file
- `MDI_API_KEY_SECRET_NAME` - Name of the AWS Secrets Manager secret containing the API key
- `AWS_REGION` - AWS region for ECR and Secrets Manager operations
- `RECIPE_DIR` - Directory containing recipe code (used by the `runc launch` command
- `HAFNIA_CLOUD` – Allow emulate cloud behaviour
- `HAFNIA_LOG` – Allow changing log level for messages 