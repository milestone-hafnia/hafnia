A command-line interface tool for managing data science experiments and resources on the MDI Platform.

## Features

- **Platform Configuration**: Easy setup and management of MDI platform settings

## Installation

## CLI Commands

### Core Commands

- `mdi configure` - Configure MDI CLI settings
- `mdi profile` - Display current configuration
- `mdi clear` - Remove stored configuration

## Configuration

The CLI tool requires initial configuration with your MDI platform credentials:

1. Run `mdi configure`
2. Enter your MDI API Key when prompted
3. Provide the MDI Platform URL
4. Verify your configuration with `mdi profile`

## Example Usage

```bash
# Configure the CLI
mdi configure

```

## Environment Variables

The CLI tool uses configuration stored in your local environment. You can view the current settings using:

```bash
mdi profile
```