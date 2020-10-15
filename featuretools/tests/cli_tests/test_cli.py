from click.testing import CliRunner

from featuretools.__main__ import info, list_primitives
import featuretools as ft


def test_info():
    runner = CliRunner()
    result = runner.invoke(info)
    assert result.exit_code == 0


def test_list_primitives():
    runner = CliRunner()
    result = runner.invoke(list_primitives)
    assert result.exit_code == 0
    for x in ft.list_primitives().columns:
        assert x in result.output
    for name in ft.list_primitives()['name']:
        assert name in result.output
