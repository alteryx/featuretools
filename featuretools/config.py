from __future__ import print_function

import logging
import os
import sys
import tempfile
from warnings import warn

import yaml


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


dirname = os.path.dirname(__file__)
default_path = os.path.join(dirname, 'config.yaml')


def _writable_dir(path):
    """Whether `path` is a directory, to which the user has write access.
    Taken from IPython source:
    https://github.com/ipython/ipython/blob/master/IPython/paths.py (`_writable_dir()`)
    """
    return os.path.isdir(path) and os.access(path, os.W_OK)


def get_featuretools_dir():
    '''Get the Featuretools directory for this platform and user.

    Uses os.path.expanduser('~') and checks for writability .
    Then adds .featuretools to the end of the path.

    Modified from IPython source:

    https://github.com/ipython/ipython/blob/master/IPython/paths.py (`get_home_dir()`)
    And
    https://github.com/ipython/ipython/blob/master/IPython/utils/path.py (`get_ipython_dir()`)
    '''
    env = os.environ
    ftdir_def = '.featuretools'

    ftdir = env.get('FEATURETOOLS_DIR', None)
    if ftdir is None:
        home_dir = os.path.expanduser('~')
        # Next line will make things work even when /home/ is a symlink to
        # /usr/home as it is on FreeBSD, for example
        home_dir = os.path.realpath(home_dir)
        if not _writable_dir(home_dir) and os.name == 'nt':
            # expanduser failed, use the registry to get the 'My Documents' folder.
            try:
                import winreg as wreg  # Py 3
            except ImportError:
                import _winreg as wreg  # Py 2
            key = wreg.OpenKey(
                wreg.HKEY_CURRENT_USER,
                "Software\Microsoft\Windows\CurrentVersion\Explorer\Shell Folders"
            )
            home_dir = wreg.QueryValueEx(key, 'Personal')[0]
            key.Close()

        ftdir = os.path.join(home_dir, ftdir_def)
    ftdir = os.path.normpath(os.path.expanduser(ftdir))

    if os.path.exists(ftdir) and not _writable_dir(ftdir):
        # ftdir exists, but is not writable
        warn("Featuretools dir '{0}' is not a writable location,"
             " using a temp directory.".format(ftdir))
        ftdir = tempfile.mkdtemp()
    elif not os.path.exists(ftdir):
        parent = os.path.dirname(ftdir)
        if not _writable_dir(parent):
            # ftdir does not exist and parent isn't writable
            warn("Featuretools dir parent '{0}' is not a writable location,"
                 " using a temp directory.".format(parent))
            ftdir = tempfile.mkdtemp()
    return ftdir


ftdir = get_featuretools_dir()
ft_config_path = os.path.join(ftdir, 'config.yaml')
csv_save_location = os.path.join(ftdir, 'csv_files')


def ensure_config_file(destination=ft_config_path):
    if not os.path.exists(destination):
        import shutil
        if not os.path.exists(os.path.dirname(destination)):
            try:
                os.mkdir(os.path.dirname(destination))
            except OSError:
                pass
        try:
            shutil.copy(default_path, destination)
        except OSError:
            eprint("Unable to copy config file. Check file permissions")


def load_config_file(path=ft_config_path):
    if not os.path.exists(path):
        path = default_path
    try:
        with open(path) as f:
            text = f.read()
            config_dict = yaml.load(text)
            return config_dict
    except OSError:
        eprint("Unable to load config file. Check file permissions")
        return {'logging': {'featuretools': 'info',
                            'featuretools.entityset': 'info',
                            'featuretools.computation_backend': 'info'}}


def ensure_data_folders():
    for dest in [csv_save_location]:
        if not os.path.exists(dest):
            try:
                os.makedirs(dest)
            except OSError:
                eprint("Unable to make folder {}. Check file permissions".format(dest))


ensure_config_file()
ensure_data_folders()
config = load_config_file()
config['csv_save_location'] = csv_save_location


def initialize_logging(config):
    loggers = config.get('logging', {})
    loggers.setdefault('featuretools', 'info')

    fmt = '%(asctime)-15s %(name)s - %(levelname)s    %(message)s'
    out_handler = logging.StreamHandler(sys.stdout)
    err_handler = logging.StreamHandler(sys.stdout)
    out_handler.setFormatter(logging.Formatter(fmt))
    err_handler.setFormatter(logging.Formatter(fmt))
    err_levels = ['WARNING', 'ERROR', 'CRITICAL']

    for name, level in list(loggers.items()):
        LEVEL = getattr(logging, level.upper())
        logger = logging.getLogger(name)
        logger.setLevel(LEVEL)
        for _handler in logger.handlers:
            logger.removeHandler(_handler)

        if level in err_levels:
            logger.addHandler(err_handler)
        else:
            logger.addHandler(out_handler)
        logger.propagate = False


initialize_logging(config)
