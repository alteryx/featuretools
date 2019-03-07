import os
import subprocess
import pytest

try:
    from builtins import reload
except Exception:
    from importlib import reload

from ..primitive_tests.test_install_primitives import pip_freeze


@pytest.fixture(scope='module')
def this_dir():
    return os.path.dirname(os.path.abspath(__file__))


@pytest.fixture(scope='module')
def plugin_to_install_dir(this_dir):
    return os.path.join(this_dir, "featuretools_plugin_tester")


def test_install_packages_from_requirements(plugin_to_install_dir):
    if 'featuretools-plugin-tester' in pip_freeze():
        subprocess.check_call(["pip", "uninstall", "-y", "featuretools-plugin-tester"])
    assert 'featuretools-plugin-tester' not in pip_freeze()

    # with pytest.raises(ImportError):
    #     from featuretools.primitives import CustomMin

    # Install plugin with entry point
    subprocess.check_call(["pip", "install", plugin_to_install_dir + '/'])
    assert 'featuretools-plugin-tester' in pip_freeze()
    # reload module
    import featuretools
    reload(featuretools.primitives)
    import featuretools_plugin_tester
    reload(featuretools)
    reload(featuretools_plugin_tester)

    # # Now plugin with primitive should work
    # from featuretools.primitives import CustomMin

    subprocess.check_call(["pip", "uninstall", "-y", "featuretools-plugin-tester"])
    assert 'featuretools-plugin-tester' not in pip_freeze()

