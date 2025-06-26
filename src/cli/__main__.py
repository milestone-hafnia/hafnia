#!/usr/bin/env python
import click

from cli import consts, dataset_cmds, experiment_cmds, profile_cmds, recipe_cmds, runc_cmds
from cli.config import Config, ConfigSchema


@click.group()
@click.pass_context
def main(ctx: click.Context) -> None:
    """Hafnia CLI."""
    ctx.obj = Config()
    ctx.max_content_width = 120


@main.command("configure")
@click.pass_obj
def configure(cfg: Config) -> None:
    """Configure Hafnia CLI settings."""

    profile_name = click.prompt("Profile Name", type=str, default=consts.DEFAULT_PROFILE_NAME)
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
    platform_url = click.prompt("Hafnia Platform URL", type=str, default=consts.DEFAULT_API_URL)
    cfg.platform_url = platform_url.strip()
    cfg.save_config()
    profile_cmds.profile_show(cfg)


@main.command("clear")
@click.pass_obj
def clear(cfg: Config) -> None:
    """Remove stored configuration."""
    cfg.clear()
    click.echo("Successfully cleared Hafnia configuration.")


main.add_command(profile_cmds.profile)
main.add_command(dataset_cmds.dataset)
main.add_command(runc_cmds.runc)
main.add_command(experiment_cmds.experiment)
main.add_command(recipe_cmds.recipe)

if __name__ == "__main__":
    main(max_content_width=120)
