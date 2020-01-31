import click
import pandas as pd
import pkg_resources

import featuretools
from featuretools.utils.cli_utils import print_info


@click.group()
def cli():
    pass


@click.command()
def info():
    print_info()


@click.command(name='list-primitives')
def list_primitives():
    try:
        with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.max_colwidth', -1, 'display.width', 1000):
            print(featuretools.list_primitives())
    except ValueError:
        with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.max_colwidth', None, 'display.width', 1000):
            print(featuretools.list_primitives())


cli.add_command(list_primitives)
cli.add_command(info)

for entry_point in pkg_resources.iter_entry_points('featuretools_cli'):
    try:
        loaded = entry_point.load()
        if hasattr(loaded, 'commands'):
            for name, cmd in loaded.commands.items():
                cli.add_command(cmd=cmd, name=name)
    except Exception:
        pass

if __name__ == "__main__":
    cli()
