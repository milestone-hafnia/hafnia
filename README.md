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