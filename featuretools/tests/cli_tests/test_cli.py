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
    with pytest.raises(SystemExit) as r:
        cli(['info'])
    assert (r.value.code == 0, r.type.__name__ == 'SystemExit')


def test_cli_list_primitives():
    with pytest.raises(SystemExit) as r:
        cli(['list-primitives'])
    assert (r.value.code == 0, r.type.__name__ == 'SystemExit')


def test_cli():
    with pytest.raises(SystemExit) as r:
        cli()
    assert (r.value.code == 0, r.type.__name__ == 'SystemExit')
