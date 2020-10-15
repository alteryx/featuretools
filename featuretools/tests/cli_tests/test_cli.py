from click.testing import CliRunner

from featuretools.__main__ import info, list_primitives


def test_info():
    runner = CliRunner()
    result = runner.invoke(info, [''])
    assert result.exit_code == 0


def test_list_primitives():
    runner = CliRunner()
    result = runner.invoke(list_primitives, [''])
    assert result.exit_code == 0
