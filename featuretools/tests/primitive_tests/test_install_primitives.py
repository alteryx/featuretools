import json
import os
import shutil
import subprocess
import tarfile

import pytest

import featuretools
from featuretools.primitives.base import PrimitiveBase
from featuretools.primitives.install import (
    extract_archive,
    get_installation_dir,
    get_installation_temp_dir,
    install_primitives,
    list_primitive_files,
    load_primitive_from_file
)

try:
    from builtins import reload
except Exception:
    from importlib import reload


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


@pytest.mark.parametrize("install_path", [
    ("primitives_to_install_dir"),
    ("amazon_path_s3"),
    ("amazon_path_http"),
    ("install_via_cli"),
    ("install_via_module"),
], indirect=True)
def test_install_primitives(install_path, primitives_to_install_dir):
    installation_dir = get_installation_dir()
    data_dir = featuretools.config.get("primitive_data_folder")
    custom_max_file = os.path.join(installation_dir, "custom_max.py")
    custom_mean_file = os.path.join(installation_dir, "custom_mean.py")
    custom_sum_file = os.path.join(installation_dir, "custom_sum.py")
    data_file = os.path.join(data_dir, "_pytest_test.csv")
    data_subfolder = os.path.join(data_dir, "pytest_folder")

    # make sure primitive files aren't there e.g from a failed run
    old_files = [custom_max_file, custom_mean_file, custom_sum_file, data_file, data_subfolder]
    remove_test_files(old_files)

    # handle install via command line as a special case
    if install_path == "INSTALL_VIA_CLI":
        subprocess.check_output(['featuretools', 'install', '--no-prompt', primitives_to_install_dir])
    elif install_path == "INSTALL_VIA_MODULE":
        subprocess.check_output(['python', '-m', 'featuretools', 'install', '--no-prompt', primitives_to_install_dir])
    else:
        install_primitives(install_path, prompt=False)

    # must reload submodule for it to work
    reload(featuretools.primitives.installed)
    from featuretools.primitives.installed import CustomMax, CustomSum, CustomMean  # noqa: F401

    files = list_primitive_files(installation_dir)
    assert {custom_max_file, custom_mean_file, custom_sum_file}.issubset(set(files))
    assert os.path.exists(data_file)
    os.unlink(data_file)
    assert os.path.exists(data_subfolder)
    assert os.path.exists(os.path.join(data_subfolder, "hello.txt"))
    shutil.rmtree(data_subfolder)

    # then delete to clean up
    for f in [custom_max_file, custom_mean_file, custom_sum_file]:
        os.unlink(f)


def test_list_primitive_files(primitives_to_install_dir):
    files = list_primitive_files(primitives_to_install_dir)
    custom_max_file = os.path.join(primitives_to_install_dir, "custom_max.py")
    custom_mean_file = os.path.join(primitives_to_install_dir, "custom_mean.py")
    custom_sum_file = os.path.join(primitives_to_install_dir, "custom_sum.py")
    assert {custom_max_file, custom_mean_file, custom_sum_file}.issubset(set(files))


def test_fails_if_data_would_be_overwritten(primitives_to_install_dir):
    installation_dir = get_installation_dir()
    data_dir = featuretools.config.get("primitive_data_folder")
    custom_max_file = os.path.join(installation_dir, "custom_max.py")
    custom_mean_file = os.path.join(installation_dir, "custom_mean.py")
    custom_sum_file = os.path.join(installation_dir, "custom_sum.py")
    data_file = os.path.join(data_dir, "_pytest_test.csv")
    data_subfolder = os.path.join(data_dir, "pytest_folder")

    # make sure primitive files aren't there e.g from a failed run
    # make sure primitive files aren't there e.g from a failed run
    old_files = [custom_max_file, custom_mean_file, custom_sum_file, data_file, data_subfolder]
    remove_test_files(old_files)

    install_primitives(primitives_to_install_dir, prompt=False)
    # should fail second time when trying to copy files that already exist
    with pytest.raises(OSError):
        install_primitives(primitives_to_install_dir, prompt=False)


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


def test_cleans_up_tmp_dir_on_error(bad_primitives_files_dir):
    tmp_dir = get_installation_temp_dir()
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)

    error_text = "(More than one primitive defined in file "\
                 "{0}/multiple_primitives\\.py|No primitive defined in file "\
                 "{0}/no_primitives\\.py)".format(bad_primitives_files_dir)
    with pytest.raises(RuntimeError, match=error_text):
        install_primitives(bad_primitives_files_dir, prompt=False)
    assert not os.path.exists(tmp_dir)


def test_errors_if_missing_primitives(primitives_to_install_dir):
    tmp_dir = get_installation_temp_dir()
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)

    info_path = os.path.join(primitives_to_install_dir, "info.json")
    with open(info_path, 'r') as f:
        old_info = json.load(f)

    new_info = {"primitives": old_info["primitives"][:]}
    new_info["primitives"].append("MissingPrimitive")
    with open(info_path, 'w') as f:
        json.dump(new_info, f)

    try:
        error_text = "Not all listed primitives discovered"
        with pytest.raises(RuntimeError, match=error_text):
            install_primitives(primitives_to_install_dir, prompt=False)
        assert not os.path.exists(tmp_dir)
    finally:
        with open(info_path, 'w') as f:
            json.dump(old_info, f)


def test_errors_if_extra_primitives(primitives_to_install_dir):
    tmp_dir = get_installation_temp_dir()
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)

    extra_path = os.path.join(primitives_to_install_dir, "extra_primitive.py")
    with open(extra_path, 'w') as f:
        f.write("from featuretools.primitives import Sum")

    try:
        error_text = "Primitive sum not listed in info.json"
        with pytest.raises(RuntimeError, match=error_text):
            install_primitives(primitives_to_install_dir, prompt=False)
        assert not os.path.exists(tmp_dir)
    finally:
        os.unlink(extra_path)


def test_extract_non_archive_errors(bad_primitives_files_dir):
    primitive_file = os.path.join(bad_primitives_files_dir, "no_primitives.py")
    error_text = "Cannot extract archive from %s. Must provide archive ending in .tar or .tar.gz" % primitive_file
    with pytest.raises(RuntimeError, match=error_text):
        extract_archive(primitive_file)


def test_install_packages_from_requirements(primitives_to_install_dir):
    def pip_freeze():
        output = subprocess.check_output(['pip', 'freeze'])
        if not isinstance(output, str):
            output = output.decode()
        return output

    # make sure dummy module isn't installed
    if "featuretools-pip-tester" in pip_freeze():
        subprocess.check_call(["pip", "uninstall", "-y", "featuretools-pip-tester"])
    assert "featuretools-pip-tester" not in pip_freeze()

    # generate requirements file with correct path
    requirements_path = os.path.join(primitives_to_install_dir, "requirements.txt")
    package_path = os.path.join(primitives_to_install_dir, "featuretools_pip_tester")
    with open(requirements_path, 'w') as f:
        f.write(package_path)

    tar_path = os.path.join(os.path.dirname(primitives_to_install_dir),
                            "test_install_primitives.tar.gz")
    if os.path.exists(tar_path):
        os.unlink(tar_path)

    with tarfile.open(tar_path, "w:gz") as tar:
        tar.add(requirements_path, arcname="requirements.txt")
        tar.add(os.path.join(primitives_to_install_dir, "info.json"),
                arcname="info.json")
        tar.add(os.path.join(primitives_to_install_dir, "custom_max.py"),
                arcname="custom_max.py")
        tar.add(os.path.join(primitives_to_install_dir, "custom_mean.py"),
                arcname="custom_mean.py")
        tar.add(os.path.join(primitives_to_install_dir, "custom_sum.py"),
                arcname="custom_sum.py")
        tar.add(os.path.join(primitives_to_install_dir, "data"), arcname="data")

    installation_dir = get_installation_dir()
    data_dir = featuretools.config.get("primitive_data_folder")
    custom_max_file = os.path.join(installation_dir, "custom_max.py")
    custom_mean_file = os.path.join(installation_dir, "custom_mean.py")
    custom_sum_file = os.path.join(installation_dir, "custom_sum.py")
    data_file = os.path.join(data_dir, "_pytest_test.csv")
    data_subfolder = os.path.join(data_dir, "pytest_folder")

    # make sure primitive files aren't there e.g from a failed run
    old_files = [custom_max_file, custom_mean_file, custom_sum_file, data_file, data_subfolder]
    remove_test_files(old_files)

    install_primitives(tar_path, prompt=False)

    # must reload submodule for it to work
    reload(featuretools.primitives.installed)
    from featuretools.primitives.installed import CustomMax, CustomSum, CustomMean  # noqa: F401

    files = list_primitive_files(installation_dir)
    assert {custom_max_file, custom_mean_file, custom_sum_file}.issubset(set(files))
    assert os.path.exists(data_file)
    os.unlink(data_file)
    os.unlink(requirements_path)
    assert os.path.exists(data_subfolder)
    assert os.path.exists(os.path.join(data_subfolder, "hello.txt"))
    shutil.rmtree(data_subfolder)
    os.unlink(tar_path)

    # then delete to clean up
    for f in [custom_max_file, custom_mean_file, custom_sum_file]:
        os.unlink(f)

    assert "featuretools-pip-tester" in pip_freeze()
    subprocess.check_call(["pip", "uninstall", "-y", "featuretools-pip-tester"])
