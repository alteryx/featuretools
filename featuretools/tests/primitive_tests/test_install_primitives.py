import os
import subprocess

import pytest

import featuretools
from featuretools.primitives.base import PrimitiveBase
from featuretools.primitives.install import (
    extract_archive,
    get_installation_dir,
    list_primitive_files,
    load_primitive_from_file
)

try:
    from builtins import reload
except Exception:
    from importlib import reload


@pytest.fixture(scope='module')
def this_dir():
    return os.path.dirname(os.path.abspath(__file__))


@pytest.fixture(scope='module')
def primitives_to_install_dir(this_dir):
    return os.path.join(this_dir, "primitives_to_install")


@pytest.fixture(scope='module')
def bad_primitives_files_dir(this_dir):
    return os.path.join(this_dir, "bad_primitive_files")


@pytest.mark.parametrize("install_path", [
    primitives_to_install_dir(this_dir()),
    os.path.join(this_dir(), "primitives_to_install.tar.gz"),
    "s3://featuretools-static/primitives_to_install.tar.gz",
    "https://s3.amazonaws.com/featuretools-static/primitives_to_install.tar.gz",
    "INSTALL_VIA_CLI",
    "INSTALL_VIA_MODULE",
])
def test_install_primitives(install_path):
    installation_dir = get_installation_dir()
    custom_max_file = os.path.join(installation_dir, "custom_max.py")
    custom_mean_file = os.path.join(installation_dir, "custom_mean.py")
    custom_sum_file = os.path.join(installation_dir, "custom_sum.py")

    # make sure primitive files aren't there e.g from a failed run
    for p in [custom_max_file, custom_mean_file, custom_sum_file]:
        try:
            os.unlink(p)
        except Exception:
            pass

    # handle install via command line as a special case
    if install_path == "INSTALL_VIA_CLI":
        subprocess.check_output(['featuretools', 'install', '--no-prompt', primitives_to_install_dir(this_dir())])
    elif install_path == "INSTALL_VIA_MODULE":
        subprocess.check_output(['python', '-m', 'featuretools', 'install', '--no-prompt', primitives_to_install_dir(this_dir())])
    else:
        featuretools.primitives.install.install_primitives(install_path, prompt=False)

    # must reload submodule for it to work
    reload(featuretools.primitives.installed)
    from featuretools.primitives.installed import CustomMax, CustomSum, CustomMean  # noqa: F401

    files = list_primitive_files(installation_dir)
    assert set(files) == {custom_max_file, custom_mean_file, custom_sum_file}

    files = list_primitive_files(installation_dir)
    # then delete to clean up
    for f in files:
        os.unlink(f)


def test_list_primitive_files(primitives_to_install_dir):
    files = list_primitive_files(primitives_to_install_dir)
    custom_max_file = os.path.join(primitives_to_install_dir, "custom_max.py")
    custom_mean_file = os.path.join(primitives_to_install_dir, "custom_mean.py")
    custom_sum_file = os.path.join(primitives_to_install_dir, "custom_sum.py")
    assert set(files) == {custom_max_file, custom_mean_file, custom_sum_file}


def test_load_primitive_from_file(primitives_to_install_dir):
    primitve_file = os.path.join(primitives_to_install_dir, "custom_max.py")
    primitive_name, primitive_obj = load_primitive_from_file(primitve_file)
    assert issubclass(primitive_obj, PrimitiveBase)


def test_errors_more_than_one_primitive_in_file(bad_primitives_files_dir):
    primitive_file = os.path.join(bad_primitives_files_dir, "multiple_primitives.py")
    error_text = 'More than one primitive defined in file %s' % primitive_file
    with pytest.raises(RuntimeError, match=error_text):
        load_primitive_from_file(primitive_file)


def test_errors_no_primitive_in_file(bad_primitives_files_dir):
    primitive_file = os.path.join(bad_primitives_files_dir, "no_primitives.py")
    error_text = 'No primitive defined in file %s' % primitive_file
    with pytest.raises(RuntimeError, match=error_text):
        load_primitive_from_file(primitive_file)


def test_extract_non_archive_errors(bad_primitives_files_dir):
    primitive_file = os.path.join(bad_primitives_files_dir, "no_primitives.py")
    error_text = "Cannot extract archive from %s. Must provide archive ending in .tar or .tar.gz" % primitive_file
    with pytest.raises(RuntimeError, match=error_text):
        extract_archive(primitive_file)
