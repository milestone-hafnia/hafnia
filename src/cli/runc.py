import click

from mdi_python_tools.platform.codebuild import handle_status, handle_mount, handle_prepare
from mdi_python_tools.platform.sagemaker import handle_launch


@click.group(name="runc")
def runc_group():
    """Experiment management commands"""
    pass


@runc_group.command(name="status")
@click.option("--verbose", type=int, default=1, help="Print verbose output")
def status(verbose: int = 0) -> None:
    """Check the status"""
    handle_status(verbose)


@runc_group.command(name="mount")
@click.argument("source", required=True)
def mount(source: str) -> None:
    """Mount user source"""
    handle_mount(source)


@runc_group.command(name="prepare")
@click.argument("source", required=True)
@click.argument("exec_cmd", required=True)
def prepare(source: str, exec_cmd: str) -> None:
    """Prepare entrypoint."""
    handle_prepare(source, exec_cmd)


@runc_group.command(name="launch")
@click.argument("task", required=True)
def launch(task: str) -> None:
    """Launch a job"""
    handle_launch(task)
