from pathlib import Path
from typing import Optional

import click

import cli.consts as consts
from cli.config import Config
from hafnia import utils


@click.group()
def dataset():
    """Manage dataset interaction"""
    pass


@dataset.command("download")
@click.argument("dataset_name")
@click.option(
    "--destination",
    "-d",
    default=None,
    required=False,
    help=f"Destination folder to save the dataset. Defaults to '{utils.PATH_DATASETS}/<dataset_name>'",
)
@click.option("--force", "-f", is_flag=True, default=False, help="Flag to enable force redownload")
@click.pass_obj
def data_download(cfg: Config, dataset_name: str, destination: Optional[click.Path], force: bool) -> Path:
    """Download dataset from Hafnia platform"""

    from hafnia.data.factory import download_or_get_dataset_path

    try:
        path_dataset = download_or_get_dataset_path(
            dataset_name=dataset_name,
            cfg=cfg,
            output_dir=destination,
            force_redownload=force,
        )
    except Exception:
        raise click.ClickException(consts.ERROR_GET_RESOURCE)
    return path_dataset
