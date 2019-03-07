import os
import subprocess
import pytest


@pytest.fixture(scope='module')
def this_dir():
    return os.path.dirname(os.path.abspath(__file__))


@pytest.fixture(scope='module')
def plugin_to_install_dir(this_dir):
    return os.path.join(this_dir, "featuretools_plugin_tester")


def install_and_import(package_dir, package_name):
    import importlib
    try:
        importlib.import_module(package_name)
    except ImportError:
        from pip._internal import main
        main(['install', package_dir])
    finally:
        globals()[package_name] = importlib.import_module(package_name)


def uninstall(package_name):
    import importlib
    try:
        from pip._internal import main
        importlib.import_module(package_name)
        main(["uninstall",  "-y", package_name])
    except ImportError:
        pass
    return


# TODO : Entry point Test
# def test_install_packages_from_requirements(plugin_to_install_dir):
#     uninstall('featuretools_plugin_tester')
#     subprocess.check_call(["pip", "uninstall", "-y", "featuretools-plugin-tester"])

#     if 'CustomMin' in globals():
#         del globals()['CustomMin']

#     # with pytest.raises(ImportError):
#     #     from featuretools.primitives import CustomMin

#     # Install plugin with entry point
#     subprocess.check_call(["pip", "install", plugin_to_install_dir + '/'])
#     install_and_import(plugin_to_install_dir + '/', 'featuretools_plugin_tester')

#     from featuretools_plugin_tester import CustomMin

#     # Now plugin with primitive should work
#     from featuretools.primitives import CustomMin
#     uninstall('featuretools_plugin_tester')
#     subprocess.check_call(["pip", "uninstall", "-y", "featuretools-plugin-tester"])

