import click


def configure() -> None:
    from mdi_python_tools.config import CONFIG, ConfigSchema
    from mdi_python_tools.platform import get_organization_id

    profile_name = click.prompt("Profile Name", type=str, default="default")
    profile_name = profile_name.strip()
    try:
        CONFIG.config_data.profiles[profile_name] = ConfigSchema()
        CONFIG.active_profile = profile_name
    except ValueError as e:
        click.echo(f"Error: {str(e)}", err=True)
        return

    api_key = click.prompt("MDI API Key", type=str, hide_input=True)
    try:
        CONFIG.api_key = api_key.strip()
    except ValueError as e:
        click.echo(f"Error: {str(e)}", err=True)
        return
    platform_url = click.prompt(
        "MDI Platform URL", type=str, default="https://api.mdi.milestonesys.com"
    )
    try:
        CONFIG.platform_url = platform_url.strip()
    except ValueError as e:
        click.echo(f"Error: {str(e)}", err=True)
        return
    try:
        CONFIG.organization_id = get_organization_id(CONFIG.get_platform_endpoint("organizations"))
    except Exception as e:
        click.echo(str(e), err=True)
        return
    CONFIG.save_config()
    active_profile()


def clear() -> None:
    from mdi_python_tools.config import CONFIG

    CONFIG.clear()
    click.echo("Successfully cleared MDI configuration.")


def profiles():
    """List all available profiles."""
    from mdi_python_tools.config import CONFIG

    profiles = CONFIG.available_profiles
    if not profiles:
        click.echo("Please configure the CLI with `mdi configure`")
        return
    active = CONFIG.active_profile

    for profile in profiles:
        status = "* " if profile == active else "  "
        print(f"{status}{profile}")

    print(f"\nActive profile: {active}")


def switch_profile(profile_name: str) -> None:
    from mdi_python_tools.config import CONFIG

    try:
        CONFIG.active_profile = profile_name
        CONFIG.save_config()
    except ValueError as e:
        click.echo(str(e))
        exit(0)
    click.echo(f"Switched to profile: {profile_name}")


def remove_profile(profile_name: str) -> None:
    from mdi_python_tools.config import CONFIG

    try:
        CONFIG.remove_profile(profile_name)
        CONFIG.save_config()
    except ValueError as e:
        click.echo(str(e))
        exit(0)
    click.echo(f"Removed profile: {profile_name}")


def active_profile() -> None:
    from rich.console import Console
    from rich.table import Table

    from mdi_python_tools.config import CONFIG

    try:
        api_key = CONFIG.api_key
    except ValueError as e:
        click.echo(str(e))
        exit(0)

    masked_key = f"{api_key[:4]}...{api_key[-4:]}" if len(api_key) > 8 else "****"
    console = Console()

    table = Table(title=f"MDI Platform Configuration: {CONFIG.active_profile}", show_header=False)
    table.add_column("Property", style="cyan")
    table.add_column("Value")

    table.add_row("API Key", masked_key)
    table.add_row("Organization", CONFIG.organization_id)
    table.add_row("Platform URL", CONFIG.platform_url)
    table.add_row("Config File", CONFIG.config_file.as_posix())
    console.print(table)
