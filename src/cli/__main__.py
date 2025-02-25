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


@main.command("profile")
def profile() -> None:
    """Display current configuration."""
    from cli import mdi_sys

    return mdi_sys.profile()


@main.command("clear")
def clear() -> None:
    """Remove stored configuration."""

    from cli import mdi_sys

    return mdi_sys.clear()


if __name__ == "__main__":
    main()
