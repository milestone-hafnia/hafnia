#!/usr/bin/env python
import click

from cli import consts, data_cmds, experiment_cmds, profile_cmds, runc_cmds
from cli.config import Config, ConfigSchema


@click.group()
@click.pass_context
def main(ctx: click.Context) -> None:
    """Hafnia CLI."""
    ctx.obj = Config()


@main.command("configure")
@click.pass_obj
def configure(cfg: Config) -> None:
    """Configure Hafnia CLI settings."""

    from hafnia.platform.api import get_organization_id

    profile_name = click.prompt("Profile Name", type=str, default="default")
    profile_name = profile_name.strip()
    try:
        cfg.add_profile(profile_name, ConfigSchema(), set_active=True)
    except ValueError:
        raise click.ClickException(consts.ERROR_CREATE_PROFILE)

    api_key = click.prompt("Hafnia API Key", type=str, hide_input=True)
    try:
        cfg.api_key = api_key.strip()
    except ValueError as e:
        click.echo(f"Error: {str(e)}", err=True)
        return
    platform_url = click.prompt("Hafnia Platform URL", type=str, default="https://api.mdi.milestonesys.com")
    cfg.platform_url = platform_url.strip()
    try:
        cfg.organization_id = get_organization_id(cfg.get_platform_endpoint("organizations"), cfg.api_key)
    except Exception:
        raise click.ClickException(consts.ERROR_ORG_ID)
    cfg.save_config()
    profile_cmds.profile_show(cfg)


@main.command("clear")
@click.pass_obj
def clear(cfg: Config) -> None:
    """Remove stored configuration."""
    cfg.clear()
    click.echo("Successfully cleared Hafnia configuration.")


main.add_command(profile_cmds.profile)
main.add_command(data_cmds.data)
main.add_command(runc_cmds.runc)
main.add_command(experiment_cmds.experiment)

if __name__ == "__main__":
    main()
