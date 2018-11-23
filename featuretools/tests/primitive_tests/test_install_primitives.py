import featuretools
# from featuretools.primitive_utils import install_primitives, PrimitiveBase, load_primitives_from_file, list_primitive_files, get_installation_dir
import os
import pytest

try:
    reload
except NameError:
    # Python 3
    from importlib import reload


@pytest.fixture(scope='module')
def primitives_to_install_dir():
    this_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(this_dir, "primitives_to_install")


def test_install_primitives(primitives_to_install_dir):
    featuretools.primitive_utils.install_primitives(primitives_to_install_dir, prompt=False)

    # check for files in install dir
    installation_dir = featuretools.primitive_utils.get_installation_dir()
    files = featuretools.primitive_utils.list_primitive_files(installation_dir)
    primitive_1_file = os.path.join(installation_dir, "primitive_1.py")
    primitive_2_file = os.path.join(installation_dir, "primitive_2.py")
    assert set(files) == {primitive_1_file, primitive_2_file}

    reload(featuretools)
    from featuretools.primitives import CustomSum

    # then delete to clean up
    for f in files:
        os.unlink(f)



def test_list_primitive_files(primitives_to_install_dir):
    files = list_primitive_files(primitives_to_install_dir)
    primitive_1_file = os.path.join(primitives_to_install_dir, "primitive_1.py")
    primitive_2_file = os.path.join(primitives_to_install_dir, "primitive_2.py")
    assert set(files) == {primitive_1_file, primitive_2_file}


def test_load_primitives_from_file(primitives_to_install_dir):
    primitive_1_file = os.path.join(primitives_to_install_dir, "primitive_1.py")
    primitive_1 = load_primitives_from_file(primitive_1_file)
    assert len(primitive_1) == 1
    assert issubclass(primitive_1["CustomSum"], PrimitiveBase)


    primitive_2_file = os.path.join(primitives_to_install_dir, "primitive_2.py")
    primitive_2 = load_primitives_from_file(primitive_2_file)
    assert len(primitive_2) == 2
    assert issubclass(primitive_2["CustomMean"], PrimitiveBase)
    assert issubclass(primitive_2["CustomMax"], PrimitiveBase)


