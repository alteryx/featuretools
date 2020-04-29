from featuretools.tests.plugin_tests.utils import (
    import_featuretools,
    install_featuretools_plugin,
    uninstall_featuretools_plugin
)


def test_plugin_warning():
    install_featuretools_plugin()
    warning = import_featuretools('warning').stdout.decode()
    debug = import_featuretools('debug').stdout.decode()
    uninstall_featuretools_plugin()

    message = 'Featuretools failed to load plugin module from library featuretools_plugin'
    traceback = 'NotImplementedError: plugin not implemented'

    assert message in warning
    assert traceback not in warning
    assert message in debug
    assert traceback in debug
