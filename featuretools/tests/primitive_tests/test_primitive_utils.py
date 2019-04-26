import os

import pytest

from featuretools.primitives.base import PrimitiveBase
from featuretools.primitives.utils import (
    get_featuretools_root,
    list_primitive_files,
    load_primitive_from_file
)


@pytest.fixture(scope='module')
def this_dir():
    return os.path.dirname(os.path.abspath(__file__))


@pytest.fixture(scope='module')
def primitives_to_install_dir(this_dir):
    return os.path.join(this_dir, "primitives_to_install")


@pytest.fixture(scope='module')
def bad_primitives_files_dir(this_dir):
    return os.path.join(this_dir, "bad_primitive_files")


def test_get_featuretools_root(this_dir):
    root = os.path.abspath(os.path.join(this_dir, '..', ".."))
    assert get_featuretools_root() == root


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
