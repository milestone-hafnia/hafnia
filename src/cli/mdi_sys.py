import click


def configure() -> None:
    from mdi_python_tools.config import CONFIG
    from mdi_python_tools.platform import get_organization_id

    api_key = click.prompt("MDI API Key", type=str, hide_input=True)
    try:
        CONFIG.set_api_key(api_key.strip())
    except ValueError as e:
        click.echo(f"Error: {str(e)}", err=True)
        return
    platform_url = click.prompt("MDI Platform URL", type=str)
    try:
        CONFIG.set_platform_url(platform_url.strip())
        CONFIG.organization_id = get_organization_id(CONFIG.get_platform_endpoint("organizations"))
    except ValueError as e:
        click.echo(f"Error: {str(e)}", err=True)
        return
    profile()


def clear() -> None:
    from mdi_python_tools.config import CONFIG

    CONFIG.clear()
    click.echo("Successfully cleared MDI configuration.")


def profile() -> None:
    from rich.console import Console
    from rich.table import Table

    from mdi_python_tools.config import CONFIG

    if not CONFIG.api_key:
        click.echo("No MDI API Key configured.")
        return
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
