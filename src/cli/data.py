from typing import Dict

import click
from rich import print as rprint


@click.group(name="data")
def data_group() -> None:
    """Data management commands"""
    pass


@data_group.command("download")
@click.argument("url")
@click.argument("destination")
def download(url: str, destination: str) -> Dict:
    """Download data to a given destination folder."""
    from mdi_python_tools.mdi_sdk import download_resource

    result = download_resource(url, destination)
    rprint(result)
