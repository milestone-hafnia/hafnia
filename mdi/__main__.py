import sys

import click
import keyring.errors

from .config import (
    ALLOWED_KEY_STORES,
    DEFAULT_API_URL,
    KEY_STORE_CONFIG,
    KEY_STORE_KEYRING,
    MDI_CONFIG_FILE,
    config,
)
from .data import load_dataset as data_load_dataset


@click.group()
def cli() -> None:
    pass


@cli.command()
@click.option("--name", help="Name of the dataset to download.")
@click.option("--force", is_flag=True, help="Force re-download of the dataset.")
def load_dataset(name: str, force: bool) -> None:
    """Load a dataset from AWS S3 bucket."""
    try:
        data_load_dataset(name, force_redownload=force)
        click.echo(f"Dataset {name} downloaded successfully.", err=True)
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@cli.command()
def login() -> None:
    """Interactive login. Create a new login profile and set it as the current."""
    while True:
        profile_name = click.prompt("Profile name", default="default")
        if config.get_profile(profile_name):
            click.echo(
                f'A profile with name "{profile_name}" already exists, please provide a'
                " different name.",
                err=True,
            )
        else:
            # If the profile name is unique, break the loop.
            break

    api_url = click.prompt("API URL", default=DEFAULT_API_URL)
    api_key = click.prompt("API key (hidden)", hide_input=True)
    key_store = click.prompt(
        "API key store",
        default=KEY_STORE_KEYRING,
        type=click.Choice(ALLOWED_KEY_STORES),
    )

    try:
        config.create_profile(profile_name, api_url, api_key, key_store)
        config.set_current_profile(profile_name)
        click.echo(
            f"Credentials stored successfully. The config file location: {MDI_CONFIG_FILE}",
            err=True,
        )
    except keyring.errors.KeyringError as e:
        click.echo(
            "Error storing api key to the keyring. "
            "In case your system does not support secure storage, "
            f"you can use '{KEY_STORE_CONFIG}' as the key store option.\nError: {e}",
            err=True,
        )
        sys.exit(1)
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


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
@click.option(
    "--api-key-file", required=True, help="Path to the file containing the API key"
)
@click.option(
    "--key-store",
    help="API key store",
    default=KEY_STORE_KEYRING,
    type=click.Choice(ALLOWED_KEY_STORES),
)
def profile_create(name: str, api_url: str, api_key_file: str, key_store: str) -> None:
    """Create a new profile."""
    try:
        with open(api_key_file) as f:
            api_key = f.read().strip()
        config.create_profile(name, api_url, api_key, key_store)
        click.echo(
            f"The profile is stored successfully. The config file location: {MDI_CONFIG_FILE}",
            err=True,
        )
    except keyring.errors.KeyringError as e:
        click.echo(
            "Error storing api key to the keyring. In case your system does not"
            f" support secure storage, you can use '{KEY_STORE_CONFIG}' as the key"
            f" store. Error: {e}",
            err=True,
        )
        sys.exit(1)
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@profile.command("update")
@click.argument("name", required=True)
@click.option("--api-url")
@click.option("--api-key-file", help="Path to the file containing the API key")
def profile_update(name: str, api_url: str, api_key_file: str) -> None:
    """Update a profile."""
    if not api_url and not api_key_file:
        click.echo(
            "At least one of the options (--api-url, --api-key-file) is required.",
            err=True,
        )
        sys.exit(1)

    try:
        if api_url:
            config.update_profile_api_url(name, api_url)
        if api_key_file:
            with open(api_key_file) as f:
                api_key = f.read().strip()
            config.update_profile_api_key(name, api_key)
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@profile.command("rm")
@click.argument("name", required=True)
def profile_rm(name: str) -> None:
    """Remove a profile."""
    try:
        config.delete_profile(name)
    except Exception as e:
        click.echo(f"Error: {e}.", err=True)
        sys.exit(1)


@profile.command("use")
@click.argument("name", required=True)
def profile_use(name: str) -> None:
    """Set a profile as current."""
    try:
        config.set_current_profile(name)
    except Exception as e:
        click.echo(f"Error: {e}.", err=True)
        sys.exit(1)


if __name__ == "__main__":
    cli()
