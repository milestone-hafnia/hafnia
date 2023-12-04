import click

from .data import load_dataset as data_load_dataset
from .data import set_credentials


@click.group()
def cli():
    pass


@cli.command()
def login():
    """Login command."""
    api_key = click.prompt("Please enter your API key")
    set_credentials(api_key)
    click.echo("API key stored successfully.")


@cli.command()
@click.option("--name", help="Name of the dataset to download.")
@click.option("--force", is_flag=True, help="Force re-download of the dataset.")
def load_dataset(name, force):
    """Load a dataset from AWS S3 bucket."""
    data_load_dataset(name, force_redownload=force)
    click.echo(f"Dataset {name} downloaded successfully.")


if __name__ == "__main__":
    cli()
