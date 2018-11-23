import os
import importlib.util
from inspect import isclass
from .primitive_base import PrimitiveBase
from builtins import input
from tqdm import tqdm
import featuretools
import shutil
from tqdm import tqdm


def install_primitives(directory, prompt=True):
    """Install primitives from the provided directory"""

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

def get_installation_dir():
    "return the path to the installation directory with in featuretools"
    featuretools_install_dir = os.path.dirname(featuretools.__file__)
    installation_dir = os.path.join(featuretools_install_dir, "primitives", "installed")
    return installation_dir


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



