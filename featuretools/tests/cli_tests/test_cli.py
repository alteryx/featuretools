from click.testing import CliRunner

from featuretools.__main__ import info, list_primitives


def test_info():
    runner = CliRunner()
    runner.invoke(info, [''])


def test_list_primitives():
    runner = CliRunner()
    runner.invoke(list_primitives, [''])
