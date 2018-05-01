import os
import tempfile
import warnings
from stat import S_IREAD

from featuretools.config import get_featuretools_dir

# TODO: how to test windows path from Unix?


def test_featuretools_dir_from_os_env():
    env = os.environ
    desired_ftdir = tempfile.mkdtemp()
    env['FEATURETOOLS_DIR'] = desired_ftdir
    ftdir = get_featuretools_dir()
    del env['FEATURETOOLS_DIR']
    assert desired_ftdir == ftdir


def test_featuretools_dir_normal():
    env = os.environ
    if 'FEATURETOOLS_DIR' in env:
        del env['FEATURETOOLS_DIR']
    assert get_featuretools_dir() == os.path.expanduser('~/.featuretools')


def test_featuretools_dir_from_os_env_not_writable():
    env = os.environ
    desired_ftdir = tempfile.mkdtemp()
    env['FEATURETOOLS_DIR'] = desired_ftdir
    os.chmod(desired_ftdir, S_IREAD)
    assert os.access(desired_ftdir, os.W_OK)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ftdir = get_featuretools_dir()
    assert desired_ftdir != ftdir and os.access(ftdir, os.W_OK)
    del env['FEATURETOOLS_DIR']
