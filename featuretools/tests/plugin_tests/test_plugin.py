from featuretools.tests.plugin_tests.utils import (
    import_featuretools,
    install_featuretools_plugin,
    uninstall_featuretools_plugin
)


def test_plugin_warning():
    install_featuretools_plugin()
    output = import_featuretools()
    warning = output.stdout.decode()
    uninstall_featuretools_plugin()

    assert 'Featuretools failed to load plugin module from library featuretools_plugin' in warning
