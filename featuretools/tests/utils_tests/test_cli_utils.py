from featuretools.utils import get_installed_packages, get_sys_info


def test_sys_info():
    sys_info = get_sys_info()
    info_keys = ["python", "python-bits", "OS",
                 "OS-release", "machine", "processor",
                 "byteorder", "LC_ALL", "LANG", "LOCALE"]
    found_keys = [k for k, _ in sys_info]
    assert set(info_keys).issubset(found_keys)


def test_installed_packages():
    installed_packages = get_installed_packages()
    requirements = ["pandas", "numpy", "tqdm", "toolz",
                    "PyYAML", "cloudpickle", "future",
                    "dask", "distributed", "psutil",
                    "Click", "scikit-learn"]
    assert set(requirements).issubset(installed_packages.keys())
