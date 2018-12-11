import click
import pandas as pd

import featuretools
from featuretools.primitives import install_primitives
from featuretools.primitives.install import get_featuretools_root


@click.group()
def cli():
    pass


@click.command()
def info():
    print("Featuretools version: %s" % featuretools.__version__)
    print("Featuretools installation directory: %s" % get_featuretools_root())


@click.command()
@click.option('--prompt/--no-prompt', default=True, help='Confirm primitives before installing')
@click.argument('directory')
def install(prompt, directory):
    install_primitives(directory, prompt)


@click.command()
def list_primitives():
    with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.max_colwidth', -1, 'display.width', 1000):
        print(featuretools.list_primitives())


cli.add_command(install)
cli.add_command(list_primitives)
cli.add_command(info)


if __name__ == "__main__":
    cli()
