import subprocess
from click.testing import CliRunner


def test_info():
    subprocess.check_output(['featuretools', 'info'])


def test_list_primitives():
    subprocess.check_output(['featuretools', 'list-primitives'])
    runner = CliRunner()
    runner.invoke(hello, ['Peter'])
