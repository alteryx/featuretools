import os
import subprocess

import pytest
from ..primitive_tests.test_install_primitives import pip_freeze, remove_test_files

import featuretools

try:
    from builtins import reload
except Exception:
    from importlib import reload


@pytest.fixture(scope='module')
def this_dir():
    return os.path.dirname(os.path.abspath(__file__))


@pytest.fixture(scope='module')
def plugin_to_install_dir(this_dir):
    return os.path.join(this_dir, "featuretools_plugin_tester")


def test_install_packages_from_requirements(plugin_to_install_dir):

    if 'featuretools-plugin-tester' in pip_freeze():
        subprocess.check_call(["pip", "uninstall", "-y",
                               'featuretools-plugin-tester'])
    assert 'featuretools-plugin-tester' not in pip_freeze()

    reload(featuretools.primitives.installed)

    # plugin has primitive we want
    text = "cannot import name 'CustomMin'"
    with pytest.raises(ImportError, match=text):
        from featuretools.primitives import CustomMin

    # install pip package
    subprocess.call(["pip", "install", plugin_to_install_dir + '/'])
    assert 'featuretools-plugin-tester' in pip_freeze()
    print(pip_freeze())
    reload(featuretools.primitives.installed)
    # Now plugin with primitive should work
    from featuretools.primitives import CustomMin

    assert 'featuretools-plugin-tester' in pip_freeze()
    subprocess.check_call(["pip", "uninstall", "-y", 'featuretools-plugin-tester'])
