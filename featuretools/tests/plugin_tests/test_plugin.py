import os
import subprocess
import sys


def enter_root_directory():
    pwd = os.path.dirname(__file__)
    join = os.path.join(pwd, '..', '..', '..')
    realpath = os.path.realpath(join)
    os.chdir(realpath)


def import_featuretools():
    return python('-c', 'import featuretools')


def install_featuretools():
    enter_root_directory()
    return python('-m', 'pip', 'install', '-e', '.')


def install_featuretools_plugin():
    enter_root_directory()
    os.chdir('./featuretools_plugin')
    return python('-m', 'pip', 'install', '-e', '.')


def python(*args):
    command = [sys.executable, *args]
    return subprocess.run(command, stderr=subprocess.PIPE)


def uninstall_featuretools_plugin():
    return python('-m', 'pip', 'uninstall', 'featuretools_plugin', '-y')


def test_plugin_warning():
    install_featuretools_plugin()
    install_featuretools()
    output = import_featuretools()
    warning = output.stderr.decode()
    uninstall_featuretools_plugin()

    assert 'Failed to load featuretools plugin from plugin library' in warning
