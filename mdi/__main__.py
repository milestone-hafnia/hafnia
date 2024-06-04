import sys

import click

from .config import DEFAULT_API_URL, config
from .data import load_dataset as data_load_dataset


@click.group()
def cli() -> None:
    pass


@cli.command()
@click.option("--name", help="Name of the dataset to download.")
@click.option("--force", is_flag=True, help="Force re-download of the dataset.")
def load_dataset(name: str, force: bool) -> None:
    """Load a dataset from AWS S3 bucket."""
    data_load_dataset(name, force_redownload=force)
    click.echo(f"Dataset {name} downloaded successfully.")


@cli.command()
def login() -> None:
    """Interactive login. Create a new login profile and set it as the current."""
    while True:
        profile_name = click.prompt("Profile name", default="default")
        if config.get_profile(profile_name):
            click.echo(
                f'A profile with name "{profile_name}" already exists, please provide a different name.'
            )
        else:
            # If the profile name is unique, break the loop.
            break

    api_url = click.prompt("API URL", default=DEFAULT_API_URL)
    api_key = click.prompt("API key")

    try:
        config.create_profile(profile_name, api_url, api_key)
        config.set_current_profile(profile_name)
    except Exception as e:
        click.echo(f"Error: {e}")
        sys.exit(1)

    click.echo("Credentials stored successfully.")


@cli.group("profile")
def profile() -> None:
    """Manage login profiles."""
    pass


@profile.command("ls")
def profile_ls() -> None:
    """List all profiles. The current profile is marked with an asterisk (*)."""
    current_profile = config.get_current_profile_name()
    name_urls = config.list_profiles()

    # We will align the output by the longest profile name.
    max_len = 0
    for name in name_urls:
        # 2 is added because we will mark the current profile with an asterisk and a space, that will increase the len by 2.
        current_len = len(name) + 2
        if current_len > max_len:
            max_len = current_len

    click.echo(f"{'  Name'.ljust(max_len)}\tURL")
    for name, url in name_urls.items():
        is_current = name == current_profile
        prefixed_name = f"* {name}" if is_current else f"  {name}"
        click.echo(f"{prefixed_name.ljust(max_len)}\t{url}")


@profile.command("create")
@click.argument("name", required=True)
@click.option("--api-url", required=True, help="API URL", default=DEFAULT_API_URL)
@click.option("--api-key", required=True, help="API key")
def profile_create(name: str, api_url: str, api_key: str) -> None:
    """Create a new profile."""
    try:
        config.create_profile(name, api_url, api_key)
    except Exception as e:
        click.echo(f"Error: {e}")
        sys.exit(1)


@profile.command("update")
@click.argument("name", required=True)
@click.option("--api-url")
@click.option("--api-key")
def profile_update(name: str, api_url: str, api_key: str) -> None:
    """Update a profile."""
    key_vals = {}
    if api_url:
        key_vals["api_url"] = api_url
    if api_key:
        key_vals["api_key"] = api_key

    if not key_vals:
        click.echo("At least one of the options (--api-url, --api-key) is required.")
        sys.exit(1)

    try:
        config.update_profile_values(name, key_vals)
    except Exception as e:
        click.echo(f"Error: {e}")
        sys.exit(1)


@profile.command("rm")
@click.argument("name", required=True)
def profile_rm(name: str) -> None:
    """Remove a profile."""
    try:
        config.delete_profile(name)
    except Exception as e:
        click.echo(f"Error: {e}.")
        sys.exit(1)


@profile.command("use")
@click.argument("name", required=True)
def profile_use(name: str) -> None:
    """Set a profile as current."""
    try:
        config.set_current_profile(name)
    except Exception as e:
        click.echo(f"Error: {e}.")
        sys.exit(1)


if __name__ == "__main__":
    cli()
