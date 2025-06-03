import click
from rich.console import Console
from rich.table import Table

import cli.consts as consts
from cli.config import Config, ConfigSchema


@click.group()
def profile():
    """Manage profile."""
    pass


@profile.command("ls")
@click.pass_obj
def profile_ls(cfg: Config) -> None:
    """List all available profiles."""
    profiles = cfg.available_profiles
    if not profiles:
        raise click.ClickException(consts.ERROR_CONFIGURE)
    active = cfg.active_profile

    for profile in profiles:
        status = "* " if profile == active else "  "
        print(f"{status}{profile}")

    print(f"\nActive profile: {active}")


@profile.command("use")
@click.argument("profile_name", required=True)
@click.pass_obj
def profile_use(cfg: Config, profile_name: str) -> None:
    """Switch to a different profile."""
    if len(cfg.available_profiles) == 0:
        raise click.ClickException(consts.ERROR_CONFIGURE)
    try:
        cfg.active_profile = profile_name
        cfg.save_config()
    except ValueError:
        raise click.ClickException(consts.ERROR_PROFILE_NOT_EXIST)
    click.echo(f"{consts.PROFILE_SWITCHED_SUCCESS} {profile_name}")


@profile.command("create")
@click.argument("api-key", required=True)
@click.option("--name", help="Specify profile name", default=consts.DEFAULT_PROFILE_NAME, show_default=True)
@click.option("--api-url", help="API URL", default=consts.DEFAULT_API_URL, show_default=True)
@click.option(
    "--activate/--no-activate", help="Activate the created profile after creation", default=True, show_default=True
)
@click.pass_obj
def profile_create(cfg: Config, name: str, api_url: str, api_key: str, activate: bool) -> None:
    """Create a new profile."""
    cfg_profile = ConfigSchema(platform_url=api_url, api_key=api_key)

    cfg.add_profile(profile_name=name, profile=cfg_profile, set_active=activate)


@profile.command("rm")
@click.argument("profile_name", required=True)
@click.pass_obj
def profile_rm(cfg: Config, profile_name: str) -> None:
    """Remove a profile."""
    if len(cfg.available_profiles) == 0:
        raise click.ClickException(consts.ERROR_CONFIGURE)

    if profile_name == cfg.active_profile:
        raise click.ClickException(consts.ERROR_PROFILE_REMOVE_ACTIVE)

    try:
        cfg.remove_profile(profile_name)
        cfg.save_config()
    except ValueError:
        raise click.ClickException(consts.ERROR_PROFILE_NOT_EXIST)
    click.echo(f"{consts.PROFILE_REMOVED_SUCCESS} {profile_name}")


@profile.command("active")
@click.pass_obj
def profile_active(cfg: Config) -> None:
    try:
        profile_show(cfg)
    except Exception as e:
        raise click.ClickException(str(e))


def profile_show(cfg: Config) -> None:
    masked_key = f"{cfg.api_key[:4]}...{cfg.api_key[-4:]}" if len(cfg.api_key) > 8 else "****"
    console = Console()

    table = Table(title=f"{consts.PROFILE_TABLE_HEADER} {cfg.active_profile}", show_header=False)
    table.add_column("Property", style="cyan")
    table.add_column("Value")

    table.add_row("API Key", masked_key)
    table.add_row("Platform URL", cfg.platform_url)
    table.add_row("Config File", cfg.config_path.as_posix())
    console.print(table)
