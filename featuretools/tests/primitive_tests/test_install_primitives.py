import os
import subprocess

import pytest

from featuretools.primitive_utils import (
    PrimitiveBase,
    get_installation_dir,
    list_primitive_files,
    load_primitives_from_file
)


@pytest.fixture(scope='module')
def primitives_to_install_dir():
    this_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(this_dir, "primitives_to_install")


@pytest.fixture(scope='module')
def primitives_to_install_archive():
    # command to make this file: tar -zcvf primitives_to_install.tar.gz primitives_to_install/*.py
    this_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(this_dir, "primitives_to_install.tar.gz")


def test_install_primitives(primitives_to_install_dir, primitives_to_install_archive):
    installation_dir = get_installation_dir()
    primitive_1_file = os.path.join(installation_dir, "primitive_1.py")
    primitive_2_file = os.path.join(installation_dir, "primitive_2.py")

    # make sure primitive files aren't there e.g from a failed run
    try:
        os.unlink(primitive_1_file)
    except:
        pass
    try:
        os.unlink(primitive_2_file)
    except:
        pass

    # test install from directory and archive
    for install_path in [primitives_to_install_dir, primitives_to_install_archive]:

        # due to how python modules are loaded/reloaded, do the installation
        # and check for installed primitives in subprocesses
        subprocess.check_output(['featuretools', "install", install_path])
        result = str(subprocess.check_output(['featuretools', "list-primitives"]))

        # make sure the custom primitives are there
        assert "custommax" in result
        assert "custommean" in result
        assert "customsum" in result

        files = list_primitive_files(installation_dir)
        assert set(files) == {primitive_1_file, primitive_2_file}

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
