import click

from mdi_python_tools.config import CONFIG


@click.group(name="sys")
def mdi_sys_group() -> None:
    """System management commands"""
    pass


@mdi_sys_group.command("configure")
def configure() -> int:
    """Configure MDI CLI settings."""
    api_key = click.prompt("MDI API Key", type=str, hide_input=True)
    try:
        CONFIG.set_api_key(api_key)
        click.echo("Successfully configured MDI CLI.")
        return 0
    except ValueError as e:
        click.echo(f"Error: {str(e)}", err=True)
        return 1


@mdi_sys_group.command("clear")
def clear() -> int:
    """Remove stored configuration."""
    CONFIG.delete_api_key()
    click.echo("Successfully cleared MDI configuration.")
    return 0


@mdi_sys_group.command("profile")
def profile() -> int:
    """Display current configuration."""
    api_key = CONFIG.get_api_key()
    if api_key:
        # Show only the first and last 4 characters of the API key
        masked_key = f"{api_key[:4]}...{api_key[-4:]}" if len(api_key) > 8 else "****"
        click.echo(f"MDI API Key: {masked_key}")
    else:
        click.echo("No MDI API Key configured.")
    return 0
