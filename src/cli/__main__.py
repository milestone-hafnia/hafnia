#!/usr/bin/env python
import click


@click.group()
def main():
    """MDI CLI."""
    pass


@main.command("configure")
def configure() -> None:
    """Configure MDI CLI settings."""
    from cli import mdi_sys

    return mdi_sys.configure()


@main.command("clear")
def clear() -> None:
    """Remove stored configuration."""

    from cli import mdi_sys

    return mdi_sys.clear()


@click.group()
def profile():
    """Manage profile."""
    pass


@profile.command("ls")
def profile_ls() -> None:
    """List all available profiles."""
    from cli import mdi_sys

    return mdi_sys.profiles()


@profile.command("use")
@click.argument("profile_name", required=True)
def profile_use(profile_name: str) -> None:
    """Switch to a different profile."""
    from cli import mdi_sys

    return mdi_sys.switch_profile(profile_name)


@profile.command("rm")
@click.argument("profile_name", required=True)
def profile_rm(profile_name: str) -> None:
    """Remove a profile."""
    from cli import mdi_sys

    return mdi_sys.remove_profile(profile_name)


@profile.command("active")
def profile_active() -> None:
    """Show active profile."""
    from cli import mdi_sys

    return mdi_sys.active_profile()


main.add_command(profile)

if __name__ == "__main__":
    main()
