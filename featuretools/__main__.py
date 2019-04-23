import click
import pandas as pd
import pkg_resources

import featuretools
from featuretools.primitives.utils import get_featuretools_root
from featuretools.utils import get_installed_packages, get_sys_info


@click.group()
def cli():
    pass


@click.command()
def info():
    print("Featuretools version: %s" % featuretools.__version__)
    print("Featuretools installation directory: %s" % get_featuretools_root())
    print("\nSYSTEM INFO")
    print("-----------")
    sys_info = get_sys_info()
    for k, stat in sys_info:
        print("{k}: {stat}".format(k=k, stat=stat))

    print("\nINSTALLED VERSIONS")
    print("------------------")
    installed_packages = get_installed_packages()
    deps = [
        ("numpy", installed_packages['numpy']),
        ("pandas", installed_packages['pandas']),
        ("tqdm", installed_packages['tqdm']),
        ("toolz", installed_packages['toolz']),
        ("PyYAML", installed_packages['PyYAML']),
        ("cloudpickle", installed_packages['cloudpickle']),
        ("future", installed_packages['future']),
        ("dask", installed_packages['dask']),
        ("distributed", installed_packages['distributed']),
        ("psutil", installed_packages['psutil']),
        ("Click", installed_packages['Click']),
        ("scikit-learn", installed_packages['scikit-learn']),
        ("pip", installed_packages['pip']),
        ("setuptools", installed_packages['setuptools']),
    ]
    for k, stat in deps:
        print("{k}: {stat}".format(k=k, stat=stat))


@click.command(name='list-primitives')
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
        pass

if __name__ == "__main__":
    cli()
