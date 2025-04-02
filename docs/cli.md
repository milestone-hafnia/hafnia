A command-line interface tool for managing data science experiments and resources on the Project Hafnia.

## Features

- **Platform Configuration**: Easy setup and management of MDI platform settings

## Installation

## CLI Commands

### Core Commands

- `mdi configure` - Configure MDI CLI settings
- `mdi clear` - Remove stored configuration
- `mdi profile` - Manage profiles (see subcommands below) 

### Profile Management

- `mdi profile ls` - List all available profiles
- `mdi profile use <profile_name>` - Switch to a different profile
- `mdi profile rm <profile_name>` - Remove a specific profile
- `mdi profile active` - Show detailed information about the active profile

### Data Management

- `mdi data get <url> <destination>` - Download resource from MDI platform to local destination

### Experiment Management

- `mdi runc launch <task>` - Launch a job within the image
- `mdi runc build <recipe_url> [state_file] [ecr_repository] [image_name]` - Build docker image with a given recipe
- `mdi runc build-local <recipe> [state_file] [image_name]` - Build recipe from local path as image with prefix - localhost

## Configuration

he CLI tool supports multiple configuration profiles:

1. Run `mdi configure`
2. Enter a profile name (defaults to "default")
3. Enter your MDI API Key when prompted
4. Provide the MDI Platform URL (defaults to "https://api.mdi.milestonesys.com")
5. The organization ID will be retrieved automatically
6. Verify your configuration with `mdi profile active`

## Example Usage

```bash
# Configure the CLI with a new profile
mdi configure

# List all available profiles
mdi profile ls

# Switch to a different profile
mdi profile use production

# View active profile details
mdi profile active

# Remove a profile
mdi profile rm old-profile

# Clear all configuration
mdi clear

# Download a dataset sample
mdi data download mnist

# Add '--force' to re-download dataset
mdi data download mnist --force

# Download a resource from the platform
mdi data get https://api.mdi.milestonesys.com/api/v1/datasets/my-dataset ./data

# Build a Docker image from a recipe
mdi runc build https://api.mdi.milestonesys.com/api/v1/recipes/my-recipe

# Build a Docker image from a local recipe
mdi runc build-local ./my-recipe

# Launch a task within the image
mdi runc launch train
```

## Environment Variables

The CLI tool uses configuration stored in your local environment. You can view the current settings using:

```bash
mdi profile active
```

Available environment variables:

- `MDI_CONFIG_PATH` - Custom path to the configuration file
- `MDI_API_KEY_SECRET_NAME` - Name of the AWS Secrets Manager secret containing the API key
- `AWS_REGION` - AWS region for ECR and Secrets Manager operations
- `RECIPE_DIR` - Directory containing recipe code (used by the `runc launch` command)