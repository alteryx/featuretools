import os
import subprocess
import sys

PWD = os.path.dirname(__file__)


def enter_root_directory():
    join = os.path.join(PWD, '..', '..', '..')
    realpath = os.path.realpath(join)
    os.chdir(realpath)


def import_featuretools():
    return python('-c', 'import featuretools')


def install_featuretools():
    enter_root_directory()
    return python('-m', 'pip', 'install', '-e', '.')


def install_plugin():
    os.chdir(PWD)
    return python('-m', 'pip', 'install', '-e', '.')


def python(*args):
    command = [sys.executable, *args]
    return subprocess.run(command, stderr=subprocess.PIPE)


def uninstall_plugin():
    return python('-m', 'pip', 'uninstall', 'featuretools_plugin', '-y')


def test_plugin_warning():
    install_plugin()
    install_featuretools()
    warning = import_featuretools()
    warning = warning.stderr.decode()\
    uninstall_plugin()

    assert 'Failed to load featuretools plugin from plugin library' in warning
