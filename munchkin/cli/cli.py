#!/usr/bin/env python
import sys

import click

import munchkin


@click.group(invoke_without_command=True)
@click.option(
    '--version/--no-version', '-v',
    default=False,
    help="Print the current version and exit."
)
@click.pass_context
def cli(ctx, version):
    """
    munchkin command line interface.
    """
    if version:
        print(munchkin.__version__)
        sys.exit(0)
