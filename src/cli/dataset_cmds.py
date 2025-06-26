from pathlib import Path
from typing import Optional

import click
import rich

import cli.consts as consts
from cli.config import Config
from hafnia import utils
from hafnia.platform.datasets import TABLE_FIELDS, create_rich_table_from_dataset


@click.group()
def dataset():
    """Manage dataset interaction"""
    pass


@dataset.command("ls")
@click.option(
    "--sort-by",
    "-s",
    default="name",
    type=click.Choice(list(TABLE_FIELDS.values()), case_sensitive=False),
    help="Sort datasets by specified field",
)
@click.pass_obj
def dataset_list(cfg: Config, sort_by: str) -> None:
    """List available datasets on Hafnia platform"""

    from hafnia.platform.datasets import dataset_list

    try:
        datasets = dataset_list(cfg=cfg)
    except Exception:
        raise click.ClickException(consts.ERROR_GET_RESOURCE)

    table = create_rich_table_from_dataset(datasets, sort_by=sort_by)
    rich.print(table)


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

    from hafnia.platform.datasets import download_or_get_dataset_path

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
