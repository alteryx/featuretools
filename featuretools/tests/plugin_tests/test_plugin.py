import os
import subprocess
import sys

PWD = os.path.dirname(__file__)


def pip(*args):
    command = [sys.executable, '-m', 'pip', *args]
    code = subprocess.run(command).returncode
    return code


def test_plugin_warning():
    pass