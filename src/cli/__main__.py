#!/usr/bin/env python
import click

from cli.data import data_group
from cli.mdi_sys import mdi_sys_group
from cli.runc import runc_group


@click.group()
def main():
    """Main CLI entry point"""
    pass


main.add_command(mdi_sys_group)
main.add_command(runc_group)
main.add_command(data_group)

if __name__ == "__main__":
    main()
