import click
from rich import print as rprint


@click.group(name="experiment")
def experiment_group() -> None:
    """System management commands"""
    pass


@experiment_group.command(name="create")
@click.argument("name")
@click.argument("source_dir")
@click.argument("exec_cmd")
@click.argument("dataset_name")
@click.argument("env_name")
def create(name: str, source_dir: str, exec_cmd: str, dataset_name: str, env_name: str) -> None:
    """Create a new experiment run"""
    from mdi_python_tools.mdi_sdk.experiment import create_experiment_run

    experiment_info = create_experiment_run(name, source_dir, exec_cmd, dataset_name, env_name)
    rprint(experiment_info)
