import click


@click.group(name="runc")
def runc_group():
    """Experiment management commands"""
    pass


@runc_group.command(name="launch")
@click.argument("task", required=True)
def launch(task: str) -> None:
    """Launch a job within the image."""
    from mdi_python_tools.platform.sagemaker import handle_launch

    handle_launch(task)


@runc_group.command(name="build")
@click.argument("recipe_url")
@click.argument("exec_cmd")
@click.argument("state_file", default="state.json")
@click.argument("ecr_repository", default="")
@click.argument("image_tag", default="")
def build(
    recipe_url: str, exec_cmd: str, state_file: str, ecr_repository: str, image_tag: str
) -> None:
    """Build docker image with a given recipe."""
    from mdi_python_tools.platform import codebuild

    codebuild.build_image(recipe_url, exec_cmd, state_file, ecr_repository, image_tag)
