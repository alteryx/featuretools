import subprocess

import pytest

from featuretools.__main__ import cli


def test_info():
    subprocess.check_output(['featuretools', 'info'])


def test_list_primitives():
    subprocess.check_output(['featuretools', 'list-primitives'])


def test_cli_help():
    subprocess.check_output(['featuretools'])


def test_cli_info():
    with pytest.raises(SystemExit):
        cli(['info'])


def test_cli_list_primitives():
    with pytest.raises(SystemExit):
        cli(['list-primitives'])


def test_cli():
    with pytest.raises(SystemExit):
        cli()
