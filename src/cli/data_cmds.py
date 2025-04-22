from typing import Optional

import click
from rich import print as rprint

import cli.consts as consts
from cli.config import Config


@click.group()
def data():
    """Manage data interaction"""
    pass


@data.command("get")
@click.argument("url")
@click.argument("destination")
@click.pass_obj
def data_get(cfg: Config, url: str, destination: click.Path) -> None:
    """Download resource from Hafnia platform"""

    from hafnia.platform import download_resource

    try:
        result = download_resource(resource_url=url, destination=str(destination), api_key=cfg.api_key)
    except Exception:
        raise click.ClickException(consts.ERROR_GET_RESOURCE)

    rprint(result)


@data.command("download")
@click.argument("dataset_name")
@click.argument("destination", default=None, required=False)
@click.option("--force", is_flag=True, default=False, help="Force download")
@click.pass_obj
def data_download(cfg: Config, dataset_name: str, destination: Optional[click.Path], force: bool) -> None:
    """Download dataset from Hafnia platform"""

    from hafnia.data.factory import download_or_get_dataset_path

    try:
        endpoint_dataset = cfg.get_platform_endpoint("datasets")
        api_key = cfg.api_key
        download_or_get_dataset_path(
            dataset_name=dataset_name,
            endpoint=endpoint_dataset,
            api_key=api_key,
            output_dir=destination,
            force_redownload=force,
        )
    except Exception:
        raise click.ClickException(consts.ERROR_GET_RESOURCE)
