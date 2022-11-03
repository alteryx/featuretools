import locale
import os
import platform
import struct
import sys

import pkg_resources

import featuretools

deps = [
    "numpy",
    "pandas",
    "tqdm",
    "cloudpickle",
    "dask",
    "distributed",
    "psutil",
    "pip",
    "setuptools",
]


def show_info():
    print("Featuretools version: %s" % featuretools.__version__)
    print("Featuretools installation directory: %s" % get_featuretools_root())
    print_sys_info()
    print_deps(deps)


def print_sys_info():
    print("\nSYSTEM INFO")
    print("-----------")
    sys_info = get_sys_info()
    for k, stat in sys_info:
        print("{k}: {stat}".format(k=k, stat=stat))


def print_deps(dependencies):
    print("\nINSTALLED VERSIONS")
    print("------------------")
    installed_packages = get_installed_packages()

    package_dep = []
    for x in dependencies:
        # prevents uninstalled deps from being printed
        if x in installed_packages:
            package_dep.append((x, installed_packages[x]))
    for k, stat in package_dep:
        print("{k}: {stat}".format(k=k, stat=stat))


# Modified from here
# https://github.com/pandas-dev/pandas/blob/d9a037ec4ad0aab0f5bf2ad18a30554c38299e57/pandas/util/_print_versions.py#L11
def get_sys_info():
    "Returns system information as a dict"

    blob = []

    try:
        (sysname, nodename, release, version, machine, processor) = platform.uname()
        blob.extend(
            [
                ("python", ".".join(map(str, sys.version_info))),
                ("python-bits", struct.calcsize("P") * 8),
                ("OS", "{sysname}".format(sysname=sysname)),
                ("OS-release", "{release}".format(release=release)),
                ("machine", "{machine}".format(machine=machine)),
                ("processor", "{processor}".format(processor=processor)),
                ("byteorder", "{byteorder}".format(byteorder=sys.byteorder)),
                ("LC_ALL", "{lc}".format(lc=os.environ.get("LC_ALL", "None"))),
                ("LANG", "{lang}".format(lang=os.environ.get("LANG", "None"))),
                ("LOCALE", ".".join(map(str, locale.getlocale()))),
            ],
        )
    except (KeyError, ValueError):
        pass

    return blob


def get_installed_packages():
    installed_packages = {}
    for d in pkg_resources.working_set:
        installed_packages[d.project_name] = d.version
    return installed_packages


def get_featuretools_root():
    return os.path.dirname(featuretools.__file__)
