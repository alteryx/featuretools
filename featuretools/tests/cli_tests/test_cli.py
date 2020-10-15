from click.testing import CliRunner

import featuretools as ft
from featuretools.__main__ import info, list_primitives


def test_info():
    runner = CliRunner()
    result = runner.invoke(info)
    assert result.exit_code == 0
    assert 'Featuretools version' in result.output
    assert 'Featuretools installation directory' in result.output


def test_list_primitives():
    runner = CliRunner()
    result = runner.invoke(list_primitives)
    assert result.exit_code == 0
    for x in ft.list_primitives().columns:
        assert x in result.output
    for name in ft.list_primitives()['name']:
        assert name in result.output
