import os
import shutil
import sys
import tarfile
from builtins import input
from inspect import isclass

import s3fs
from botocore.exceptions import NoCredentialsError
from smart_open import smart_open
from tqdm import tqdm

from .base.primitive_base import PrimitiveBase

import featuretools

IS_PY2 = (sys.version_info[0] == 2)


if IS_PY2:
    import imp
else:
    import importlib.util

if IS_PY2:
    from six.moves.urllib.parse import urlparse
else:
    from urllib.parse import urlparse


def install_primitives(directory_or_archive, prompt=True):
    """Install primitives from the provided directory"""
    tmp_dir = get_installation_temp_dir()

    # if it isn't local, download it. if remote, it must be archive
    if not (os.path.isdir(directory_or_archive) or os.path.isfile(directory_or_archive)):
        directory_or_archive = download_archive(directory_or_archive)

    # if archive, extract directory to temp folders
    if os.path.isfile(directory_or_archive):
        directory = extract_archive(directory_or_archive)
    else:
        directory = directory_or_archive

    # Iterate over all the files and determine the primitives to install
    files = list_primitive_files(directory)
    all_primitives = {}
    files_to_copy = []
    for filepath in files:
        primitive_name, primitive_obj = load_primitive_from_file(filepath)
        files_to_copy.append(filepath)
        all_primitives[primitive_name] = primitive_obj

    # before installing, confirm with user
    primitives_list = ", ".join(all_primitives.keys())
    if prompt:
        while True:
            resp = input("Install primitives: %s? (Y/n) " % primitives_list)
            if resp == "Y":
                break
            elif resp == "n":
                return
    else:
        print("Installing primitives: %s" % primitives_list)

    # copy the files
    installation_dir = get_installation_dir()
    for to_copy in tqdm(files_to_copy):
        shutil.copy2(to_copy, installation_dir)

    # clean up tmp dir
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)


def get_featuretools_root():
    return os.path.dirname(featuretools.__file__)


def get_installation_dir():
    "return the path to the installation directory with in featuretools"
    installation_dir = os.path.join(get_featuretools_root(), "primitives", "installed")
    return installation_dir


def get_installation_temp_dir():
    """Returns the path to the installation directory with in featuretools.

        If the directory, doesn't exist it is created
    """
    tmp_dir = os.path.join(get_installation_dir(), ".tmp/")
    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir)
    return os.path.join(get_installation_dir(), ".tmp/")


def download_archive(uri):
    # determine where to save locally
    filename = os.path.basename(urlparse(uri).path)
    local_archive = os.path.join(get_installation_temp_dir(), filename)

    with open(local_archive, 'wb') as f:
        try:
            remote_archive = smart_open(uri, 'rb', ignore_extension=True)
        except NoCredentialsError:
            # fallback to anonymous using s3fs
            s3 = s3fs.S3FileSystem(anon=True)
            remote_archive = s3.open(uri, 'rb')

        for line in remote_archive:
            f.write(line)

        remote_archive.close()

    return local_archive


def extract_archive(filepath):
    if (filepath.endswith("tar.gz")):
        tar = tarfile.open(filepath, mode='r:gz')
    elif (filepath.endswith("tar")):
        tar = tarfile.open(filepath, "r:")
    else:
        e = "Cannot extract archive from %s." % filepath
        e += " Must provide archive ending in .tar or .tar.gz"
        raise RuntimeError(e)

    tmp_dir = get_installation_temp_dir()
    members = [m for m in tar.getmembers() if check_valid_primitive_path(m.path)]
    tar.extractall(tmp_dir, members=members)
    tar.close()

    # figure out the directory name from any file in archive
    directory = os.path.join(tmp_dir, os.path.dirname(members[0].path))

    return directory


def list_primitive_files(directory):
    """returns list of files in directory that might contain primitives"""
    files = os.listdir(directory)
    keep = []
    for path in files:
        if not check_valid_primitive_path(path):
            continue
        keep.append(os.path.join(directory, path))
    return keep


def check_valid_primitive_path(path):
    if os.path.isdir(path):
        return False

    filename = os.path.basename(path)

    if filename[:2] == "__" or filename[0] == "." or filename[-3:] != ".py":
        return False

    return True


def load_primitive_from_file(filepath):
    """load primitive objects in a file"""
    module = os.path.basename(filepath)[:-3]
    if IS_PY2:
        # for python 2.7
        module = imp.load_source(module, filepath)
    else:
        # TODO: what is the first argument"?
        # for python >3.5
        spec = importlib.util.spec_from_file_location(module, filepath)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

    primitives = []
    for primitive_name in vars(module):
        primitive_class = getattr(module, primitive_name)
        if isclass(primitive_class) and issubclass(primitive_class, PrimitiveBase):
            primitives.append((primitive_name, primitive_class))

    if len(primitives) == 0:
        raise RuntimeError("No primitive defined in file %s" % filepath)
    elif len(primitives) > 1:
        raise RuntimeError("More than one primitive defined in file %s" % filepath)

    return primitives[0]
