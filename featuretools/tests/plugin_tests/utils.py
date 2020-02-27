import os
import subprocess
import sys

import pandas as pd


def import_featuretools():
    return python('-c', 'import featuretools')


def install_featuretools_plugin():
    pwd = os.path.dirname(__file__)
    os.chdir(os.path.join(pwd, 'featuretools_plugin'))
    return python('-m', 'pip', 'install', '-e', '.')


def python(*args):
    command = [sys.executable, *args]
    return subprocess.run(command, stderr=subprocess.PIPE)


def reinstall_pandas():
    pandas = 'pandas==%s' % pd.__version__
    return python('-m', 'pip', 'install', pandas)


def uninstall_featuretools_plugin():
    return python('-m', 'pip', 'uninstall', 'featuretools_plugin', '-y')
