#!/usr/bin/env python
import click

from cli import mdi_sys
from cli.data import data_group
from cli.experiment import experiment_group
from cli.runc import runc_group


@click.group()
def main():
    """Main CLI entry point"""
    pass


@main.command("configure")
def configure() -> int:
    """Configure MDI CLI settings."""
    return mdi_sys.configure()


@main.command("profile")
def profile() -> int:
    """Display current configuration."""
    return mdi_sys.profile()


@main.command("clear")
def clear() -> int:
    """Remove stored configuration."""
    return mdi_sys.clear()


main.add_command(runc_group)
main.add_command(data_group)
main.add_command(experiment_group)

if __name__ == "__main__":
    main()
