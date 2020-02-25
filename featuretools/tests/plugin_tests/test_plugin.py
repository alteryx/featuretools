import os
import subprocess
import sys


def enter_root_directory():
    join = os.path.join('..', '..', '..')
    realpath = os.path.realpath(join)
    os.chdir(realpath)


def python(*args):
    command = [sys.executable, *args]
    code = subprocess.run(command).returncode
    return code


def import_featuretools():
    python('-c', 'import featuretools')


def install_plugin():
    python('-m', 'pip', 'install', '.')


def uninstall_plugin():
    python('-m', 'pip', 'uninstall', 'featuretools_plugin', '-y')


def test_plugin_warning():
    install_plugin()
    enter_root_directory()
    import_featuretools()
    uninstall_plugin()
