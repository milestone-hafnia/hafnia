import click

from mdi_python_tools.platform.codebuild import run_build
from mdi_python_tools.platform.sagemaker import handle_launch


@click.group(name="runc")
def runc_group():
    """Experiment management commands"""
    pass


@runc_group.command(name="launch")
@click.argument("task", required=True)
def launch(task: str) -> None:
    """Launch a job"""
    handle_launch(task)


@runc_group.command(name="run_build")
@click.argument("recipe_url")
@click.argument("exec_cmd")
@click.argument("state_file", default="state.json")
@click.argument("ecr_repository", default="")
def build(recipe_url: str, exec_cmd: str, state_file: str, ecr_repository: str) -> None:
    """Build docker image with a given recipe."""
    run_build(recipe_url, exec_cmd, state_file, ecr_repository)


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
