import os
import subprocess
import sys


def import_featuretools(level=None):
    c = ''
    if level:
        c += 'import os;'
        c += 'os.environ["FEATURETOOLS_LOG_LEVEL"] = "%s";' % level

    c += 'import featuretools;'
    return python('-c', c)


def install_featuretools_plugin():
    pwd = os.path.dirname(__file__)
    os.chdir(os.path.join(pwd, 'featuretools_plugin'))
    return python('-m', 'pip', 'install', '-e', '.')


def python(*args):
    command = [sys.executable, *args]
    return subprocess.run(command, stdout=subprocess.PIPE)


def uninstall_featuretools_plugin():
    return python('-m', 'pip', 'uninstall', 'featuretools_plugin', '-y')
