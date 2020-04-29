import os

import pytest

from featuretools.utils import (
    get_featuretools_root,
    get_installed_packages,
    get_sys_info
)


@pytest.fixture
def this_dir():
    return os.path.dirname(os.path.abspath(__file__))


def test_sys_info():
    sys_info = get_sys_info()
    info_keys = ["python", "python-bits", "OS",
                 "OS-release", "machine", "processor",
                 "byteorder", "LC_ALL", "LANG", "LOCALE"]
    found_keys = [k for k, _ in sys_info]
    assert set(info_keys).issubset(found_keys)


def test_installed_packages():
    installed_packages = get_installed_packages()
    # Per PEP 426, package names are case insensitive
    # Underscore and hyphen are equivalent
    installed_set = {name.lower().replace('-', '_')
                     for name in installed_packages.keys()}
    requirements = ["pandas", "numpy", "tqdm",
                    "pyyaml", "cloudpickle",
                    "dask", "distributed", "psutil",
                    "click"]
    assert set(requirements).issubset(installed_set)


def test_get_featuretools_root(this_dir):
    root = os.path.abspath(os.path.join(this_dir, '..', ".."))
    assert get_featuretools_root() == root
