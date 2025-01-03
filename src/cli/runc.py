import click

from mdi_python_tools.platform.codebuild import run_build

# from mdi_python_tools.platform.runtime import (
#     handle_mount,
#     handle_prepare,
#     handle_status,
# )
from mdi_python_tools.platform.sagemaker import handle_launch


@click.group(name="runc")
def runc_group():
    """Experiment management commands"""
    pass


# @runc_group.command(name="status")
# @click.option("--verbose", type=int, default=1, help="Print verbose output")
# def status(verbose: int = 0) -> None:
#     """Check the status"""
#     handle_status(verbose)


# @runc_group.command(name="mount")
# @click.argument("source", required=True)
# def mount(source: str) -> None:
#     """Mount user source"""
#     handle_mount(source)


# @runc_group.command(name="prepare")
# @click.argument("source", required=True)
# @click.argument("exec_cmd", required=True)
# def prepare(source: str, exec_cmd: str) -> None:
#     """Prepare entrypoint."""
# handle_prepare(source, exec_cmd)


@runc_group.command(name="launch")
@click.argument("task", required=True)
def launch(task: str) -> None:
    """Launch a job"""
    handle_launch(task)


@runc_group.command(name="run_build")
@click.argument("recipe_url")
def build(recipe_url: str) -> None:
    """Build docker image with a given recipe."""
    run_build(recipe_url)


# def list_training_runs():
#     """List training runs."""
#     api_key = _ensure_api_key()
#     headers = headers_from_api_key(api_key)
#     url = f"{config.get_api_url()}/api/v1/training-runs"
#     r = requests.get(url, headers=headers)
#     if r.status_code == 200:
#         return r.json()
#     else:
#         r.raise_for_status()


# def create_training_run(name: str, description: str, file):
#     api_key = _ensure_api_key()
#     headers = headers_from_api_key(api_key)
#     url = f"{config.get_api_url()}/api/v1/training-runs"
#     body = {"model_name": name, "description": description}
#     r = requests.post(url, headers=headers, data=body, files=dict(recipe=file))
#     if r.status_code == 201:
#         return r.json()
#     else:
#         r.raise_for_status()
