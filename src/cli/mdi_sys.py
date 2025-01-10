import click

from mdi_python_tools.config import CONFIG
from mdi_python_tools.platform import get_organization_id


def configure() -> int:
    api_key = click.prompt("MDI API Key", type=str, hide_input=True)
    try:
        CONFIG.set_api_key(api_key)
    except ValueError as e:
        click.echo(f"Error: {str(e)}", err=True)
        return 1
    platform_url = click.prompt("MDI Platform URL", type=str)
    try:
        CONFIG.set_platform_url(platform_url)
        CONFIG.organization_id = get_organization_id(CONFIG.get_platform_endpoint("organizations"))
    except ValueError as e:
        click.echo(f"Error: {str(e)}", err=True)
        return 1
    profile()


def clear() -> int:
    CONFIG.clear()
    click.echo("Successfully cleared MDI configuration.")
    return 0


def profile() -> int:
    from rich.console import Console
    from rich.table import Table

    if not CONFIG.api_key:
        click.echo("No MDI API Key configured.")
        return 0
    masked_key = (
        f"{CONFIG.api_key[:4]}...{CONFIG.api_key[-4:]}" if len(CONFIG.api_key) > 8 else "****"
    )
    console = Console()

    table = Table(title="MDI Platform Configuration", show_header=False)
    table.add_column("Property", style="cyan")
    table.add_column("Value")

    table.add_row("API Key", masked_key)
    table.add_row("Organization", CONFIG.organization_id)
    table.add_row("Platform URL", CONFIG.platform_url)
    table.add_row("Config File", CONFIG.config_file.as_posix())
    console.print(table)
    return 0
