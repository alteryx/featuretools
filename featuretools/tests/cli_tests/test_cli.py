import pytest
import subprocess
from featuretools.__main__ import cli


def test_info():
    assert subprocess.check_output(['featuretools', 'info'], stderr=subprocess.STDOUT).startswith(b'Featuretools')


def test_list_primitives():
    subprocess.check_output(['featuretools', 'list-primitives'])


def test_cli_help():
    help_msg = (b'''Usage: featuretools [OPTIONS] COMMAND [ARGS]...

Options:
  --help  Show this message and exit.

Commands:
  info
  list-primitives
''')
    assert subprocess.check_output(['featuretools'], stderr=subprocess.STDOUT) == help_msg


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

