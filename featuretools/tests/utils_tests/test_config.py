import logging
import os

from featuretools.config_init import initialize_logging

logging_env_vars = {'FEATURETOOLS_LOG_LEVEL': "debug",
                    'FEATURETOOLS_ES_LOG_LEVEL': "critical",
                    'FEATURETOOLS_BACKEND_LOG_LEVEL': "error"}


def test_logging_defaults():
    old_env_vars = {}
    for env_var in logging_env_vars:
        old_env_vars[env_var] = os.environ.get(env_var, None)
        if old_env_vars[env_var] is not None:
            del os.environ[env_var]

    initialize_logging()
    main_logger = logging.getLogger('featuretools')
    assert main_logger.getEffectiveLevel() == logging.INFO
    es_logger = logging.getLogger('featuretools.entityset')
    assert es_logger.getEffectiveLevel() == logging.INFO
    backend_logger = logging.getLogger('featuretools.computation_backend')
    assert backend_logger.getEffectiveLevel() == logging.INFO

    for env_var, value in old_env_vars.items():
        if value is not None:
            os.environ[env_var] = value


def test_logging_set_via_env():
    old_env_vars = {}
    for env_var, value in logging_env_vars.items():
        old_env_vars[env_var] = os.environ.get(env_var, None)
        os.environ[env_var] = value

    initialize_logging()
    main_logger = logging.getLogger('featuretools')
    assert main_logger.getEffectiveLevel() == logging.DEBUG
    es_logger = logging.getLogger('featuretools.entityset')
    assert es_logger.getEffectiveLevel() == logging.CRITICAL
    backend_logger = logging.getLogger('featuretools.computation_backend')
    assert backend_logger.getEffectiveLevel() == logging.ERROR

    for env_var, value in old_env_vars.items():
        if value is not None:
            os.environ[env_var] = value
