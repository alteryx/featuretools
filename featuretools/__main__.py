import click
import pandas as pd
import pkg_resources

import featuretools
from featuretools.primitives.utils import get_featuretools_root


@click.group()
def cli():
    pass


@click.command()
def info():
    print("Featuretools version: %s" % featuretools.__version__)
    print("Featuretools installation directory: %s" % get_featuretools_root())


@click.command()
def list_primitives():
    with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.max_colwidth', -1, 'display.width', 1000):
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
        raise
        pass

if __name__ == "__main__":
    cli()
