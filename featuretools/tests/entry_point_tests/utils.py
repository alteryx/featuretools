import os
import subprocess
import sys


def get_relative_path(*args):
    pwd = os.path.dirname(__file__)
    return os.path.join(pwd, *args)


def python(*args):
    command = [sys.executable, *args]
    return subprocess.run(command, stdout=subprocess.PIPE)


def install_featuretools_plugin():
    os.chdir(get_relative_path('add-ons', 'featuretools_plugin'))
    return python('-m', 'pip', 'install', '-e', '.')


def uninstall_featuretools_plugin():
    return python('-m', 'pip', 'uninstall', 'featuretools_plugin', '-y')


def install_featuretools_primitives():
    os.chdir(get_relative_path('add-ons', 'featuretools_primitives'))
    return python('-m', 'pip', 'install', '-e', '.')


def uninstall_featuretools_primitives():
    return python('-m', 'pip', 'uninstall', 'featuretools_primitives', '-y')


def import_featuretools(level=None):
    c = ''
    if level:
        c += 'import os;'
        c += 'os.environ["FEATURETOOLS_LOG_LEVEL"] = "%s";' % level

    c += 'import featuretools;'
    return python('-c', c)
