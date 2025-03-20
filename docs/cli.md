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

## Configuration

he CLI tool supports multiple configuration profiles:

1. Run mdi configure
2. Enter a profile name (defaults to "default")
3. Enter your MDI API Key when prompted
4. Provide the MDI Platform URL (defaults to "https://api.mdi.milestonesys.com")
5. The organization ID will be retrieved automatically
6. Verify your configuration with mdi profile active

## `Example Usage

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

```

## Environment Variables

The CLI tool uses configuration stored in your local environment. You can view the current settings using:

```bash
mdi profile active
```