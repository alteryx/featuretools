from featuretools.tests.plugin_tests.utils import (
    import_featuretools,
    install_featuretools_plugin,
    uninstall_featuretools_plugin
)


def test_plugin_warning():
    install_featuretools_plugin()
    output = import_featuretools()
    warning = output.stderr.decode()
    uninstall_featuretools_plugin()

    assert 'Failed to load featuretools plugin from plugin library' in warning
