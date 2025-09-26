from pathlib import Path
from typing import Optional

import click

import cli.consts as consts
from cli.config import Config


@click.group(name="trainer")
def trainer_package() -> None:
    """Trainer package commands"""
    pass


@trainer_package.command(name="ls")
@click.pass_obj
@click.option("-l", "--limit", type=int, default=None, help="Limit number of listed trainer packages.")
def cmd_list_trainer_packages(cfg: Config, limit: Optional[int]) -> None:
    """List available trainer packages on the platform"""

    from hafnia.platform.trainer_package import get_trainer_packages, pretty_print_trainer_packages

    endpoint = cfg.get_platform_endpoint("trainers")
    trainers = get_trainer_packages(endpoint, cfg.api_key)

    pretty_print_trainer_packages(trainers, limit=limit)


@trainer_package.command(name="create-zip")
@click.argument("source")
@click.option(
    "--output",
    type=click.Path(writable=True),
    default="./trainer.zip",
    show_default=True,
    help="Output trainer package path.",
)
def cmd_create_trainer_package_zip(source: str, output: str) -> None:
    """Create Hafnia trainer package as zip-file from local path"""

    from hafnia.utils import archive_dir

    path_output_zip = Path(output)
    if path_output_zip.suffix != ".zip":
        raise click.ClickException(consts.ERROR_TRAINER_PACKAGE_FILE_FORMAT)

    path_source = Path(source)
    path_output_zip = archive_dir(path_source, path_output_zip)


@trainer_package.command(name="view-zip")
@click.option("--path", type=str, default="./trainer.zip", show_default=True, help="Path of trainer.zip.")
@click.option("--depth-limit", type=int, default=3, help="Limit the depth of the tree view.", show_default=True)
def cmd_view_trainer_package_zip(path: str, depth_limit: int) -> None:
    """View the content of a trainer package zip file."""
    from hafnia.utils import show_trainer_package_content

    path_trainer_package = Path(path)
    if not path_trainer_package.exists():
        raise click.ClickException(
            f"Trainer package file '{path_trainer_package}' does not exist. Please provide a valid path. "
            f"To create a trainer package, use the 'hafnia trainer create-zip' command."
        )
    show_trainer_package_content(path_trainer_package, depth_limit=depth_limit)
