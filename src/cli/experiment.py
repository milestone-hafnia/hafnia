import click
from rich import print as rprint

from mdi_python_tools.mdi_sdk.experiment import create_experiment_run


@click.group(name="experiment")
def experiment_group() -> None:
    """System management commands"""
    pass


@experiment_group.command(name="create")
@click.argument("source_dir")
@click.argument("exec_cmd")
@click.argument("dataset_name")
def create(source_dir: str, exec_cmd: str, dataset_name: str) -> None:
    experiment_info = create_experiment_run(source_dir, exec_cmd, dataset_name)
    rprint(experiment_info)
