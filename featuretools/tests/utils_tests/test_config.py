import os
# import stat
# import subprocess
import tempfile

from featuretools.config import get_featuretools_dir

# import warnings


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


# TODO: this doesn't work on CircleCI because it runs tox as the root user
# def test_featuretools_dir_from_os_env_not_writable():
#     env = os.environ
#     desired_ftdir = tempfile.mkdtemp()
#     env['FEATURETOOLS_DIR'] = desired_ftdir
#     subprocess.call(["chattr", str(stat.SF_IMMUTABLE), desired_ftdir])
#     os.chmod(desired_ftdir, stat.S_IREAD)
#     assert not os.access(desired_ftdir, os.W_OK)
#     with warnings.catch_warnings():
#         warnings.simplefilter("ignore")
#         ftdir = get_featuretools_dir()
#     assert desired_ftdir != ftdir and os.access(ftdir, os.W_OK)
#     del env['FEATURETOOLS_DIR']
