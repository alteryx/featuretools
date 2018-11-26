import click
import pandas as pd

import featuretools
from featuretools.primitives import install_primitives


@click.group()
def cli():
    pass


@click.command()
@click.option('--prompt', default=True, help='Confirm primitives before installing')
@click.argument('directory')
def install(prompt, directory):
    install_primitives(directory, prompt)


@click.command()
def list_primitives():
    # pd.set_option('display.max_colwidth', -1)
    with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.max_colwidth', -1, 'display.width', 1000):
        print(featuretools.list_primitives())


cli.add_command(install)
cli.add_command(list_primitives)
