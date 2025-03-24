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
    """Download resource from MDI platform"""

    from mdi_python_tools.data.s3_client import download_resource

    try:
        result = download_resource(url, destination, cfg.api_key)
    except Exception:
        raise click.ClickException(consts.ERROR_GET_RESOURCE)

    rprint(result)
