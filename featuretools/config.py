import logging
import os
import sys

import yaml

dirname = os.path.dirname(__file__)
default_path = os.path.join(dirname, 'config_yaml.txt')
ft_config_path = os.path.join(os.path.expanduser('~'), '.featuretools', 'config_yaml.txt')
csv_save_location = os.path.join(os.path.expanduser('~'), '.featuretools', 'csv_files')


def ensure_config_file(destination=ft_config_path):
    if not os.path.exists(destination):
        import shutil
        if not os.path.exists(os.path.dirname(destination)):
            try:
                os.mkdir(os.path.dirname(destination))
            except OSError:
                pass
        shutil.copy(default_path, destination)


def load_config_file(path=ft_config_path):
    if not os.path.exists(path):
        path = default_path
    with open(path) as f:
        text = f.read()
        config_dict = yaml.load(text)
        return config_dict


def ensure_data_folders():
    for dest in [csv_save_location]:
        if not os.path.exists(dest):
            os.makedirs(dest)


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
