import os
import subprocess
import sys


def _get_path_to_add_ons(*args):
    pwd = os.path.dirname(__file__)
    return os.path.join(pwd, "add-ons", *args)


def _python(*args):
    command = [sys.executable, *args]
    return subprocess.run(command, stdout=subprocess.PIPE)


def _install_featuretools_plugin():
    os.chdir(_get_path_to_add_ons("featuretools_plugin"))
    return _python("-m", "pip", "install", "-e", ".")


def _uninstall_featuretools_plugin():
    return _python("-m", "pip", "uninstall", "featuretools_plugin", "-y")


def _install_featuretools_primitives():
    os.chdir(_get_path_to_add_ons("featuretools_primitives"))
    return _python("-m", "pip", "install", "-e", ".")


def _uninstall_featuretools_primitives():
    return _python("-m", "pip", "uninstall", "featuretools_primitives", "-y")


def _import_featuretools(level=None):
    c = ""
    if level:
        c += "import os;"
        c += 'os.environ["FEATURETOOLS_LOG_LEVEL"] = "%s";' % level

    c += "import featuretools;"
    return _python("-c", c)
