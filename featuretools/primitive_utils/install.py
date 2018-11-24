import importlib.util
import os
import shutil
import tarfile
from builtins import input
from inspect import isclass

from tqdm import tqdm

from .primitive_base import PrimitiveBase

import featuretools


def install_primitives(directory_or_archive, prompt=True):
    """Install primitives from the provided directory"""
    tmp_dir = get_installation_temp_dir()

    # if archive, extract directory to temp folders
    if not os.path.isdir(directory_or_archive):
        if (directory_or_archive.endswith("tar.gz")):
            tar = tarfile.open(directory_or_archive, mode='r:gz')
            tar.extractall(tmp_dir)
        elif (directory_or_archive.endswith("tar")):
            tar = tarfile.open(directory_or_archive, "r:")
            tar.extractall(tmp_dir)

        # figure out the directory name from any file in archive
        directory = os.path.dirname(tar.getnames()[0])

        tar.close()
    else:
        directory = directory_or_archive

    # Iterate over all the files and determine how the primitives
    # to install
    files = list_primitive_files(directory)
    all_primitives = {}
    files_to_copy = []
    for filepath in files:
        primitives = load_primitives_from_file(filepath)
        # TODO: check if primitive is already installed. if it is, don't try to reinstall
        if len(primitives):
            all_primitives.update(primitives)
            files_to_copy.append(filepath)

    # before installing, confirm with user
    if prompt:
        resp = input("Install %d primitives? (Y/n)" % len(all_primitives))
        if resp != "Y":
            return
    else:
        print("Installing %d primitives" % len(all_primitives))

    # copy the files
    installation_dir = get_installation_dir()
    for to_copy in tqdm(files_to_copy):
        shutil.copy2(to_copy, installation_dir)

    try:
        shutil.rmtree(tmp_dir)
    except FileNotFoundError:
        pass


def get_installation_dir():
    "return the path to the installation directory with in featuretools"
    featuretools_install_dir = os.path.dirname(featuretools.__file__)
    installation_dir = os.path.join(featuretools_install_dir, "primitives", "installed")
    return installation_dir


def get_installation_temp_dir():
    "return the path to the installation directory with in featuretools"
    return os.path.join(get_installation_dir(), ".tmp/")


def list_primitive_files(directory):
    """returns list of files in directory that might contain primitives"""
    files = os.listdir(directory)
    keep = []
    for path in files:
        if path[:2] == "__" or path[0] == "." or path[-3:] != ".py":
            continue
        keep.append(os.path.join(directory, path))
    return keep


def load_primitives_from_file(filepath):
    """load primitive objects in a file"""
    # TODO: what is "module.name"?
    spec = importlib.util.spec_from_file_location("module.name", filepath)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    primitives = {}
    for primitive_name in dir(module):
        primitive_class = getattr(module, primitive_name)
        if isclass(primitive_class) and issubclass(primitive_class, PrimitiveBase):
            primitives[primitive_name] = primitive_class

    return primitives
