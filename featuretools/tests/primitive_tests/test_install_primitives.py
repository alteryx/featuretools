import os
import shutil

import pytest

from featuretools.primitives.base import PrimitiveBase
from featuretools.primitives.install import (
    list_primitive_files,
    load_primitive_from_file
)


def remove_test_files(file_list):
    # make sure primitive files aren't there e.g from a failed run
    for p in file_list:
        try:
            if os.path.isdir(p):
                shutil.rmtree(p)
            else:
                os.unlink(p)
        except Exception:
            pass


@pytest.fixture(scope='module')
def this_dir():
    return os.path.dirname(os.path.abspath(__file__))


@pytest.fixture(scope='module')
def primitives_to_install_dir(this_dir):
    return os.path.join(this_dir, "primitives_to_install")


@pytest.fixture(scope='module')
def primitives_to_install_gzip(this_dir):
    return os.path.join(this_dir, "primitives_to_install.tar.gz")


@pytest.fixture(scope='module')
def bad_primitives_files_dir(this_dir):
    return os.path.join(this_dir, "bad_primitive_files")


@pytest.fixture
def amazon_path_s3():
    return "s3://featuretools-static/primitives_to_install_v1.tar.gz"


@pytest.fixture
def amazon_path_http():
    return "https://s3.amazonaws.com/featuretools-static/primitives_to_install_v1.tar.gz"


@pytest.fixture
def install_via_cli():
    return "INSTALL_VIA_CLI"


@pytest.fixture
def install_via_module():
    return "INSTALL_VIA_MODULE"


@pytest.fixture
def install_path(request):
    return request.getfixturevalue(request.param)


def test_list_primitive_files(primitives_to_install_dir):
    files = list_primitive_files(primitives_to_install_dir)
    custom_max_file = os.path.join(primitives_to_install_dir, "custom_max.py")
    custom_mean_file = os.path.join(primitives_to_install_dir, "custom_mean.py")
    custom_sum_file = os.path.join(primitives_to_install_dir, "custom_sum.py")
    assert {custom_max_file, custom_mean_file, custom_sum_file}.issubset(set(files))


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
