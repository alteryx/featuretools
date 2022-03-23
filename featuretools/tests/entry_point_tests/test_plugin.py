from featuretools.tests.entry_point_tests.utils import (
    _import_featuretools,
    _install_featuretools_plugin,
    _uninstall_featuretools_plugin,
)


def test_plugin_warning():
    _install_featuretools_plugin()
    warning = _import_featuretools("warning").stdout.decode()
    debug = _import_featuretools("debug").stdout.decode()
    _uninstall_featuretools_plugin()

    message = (
        "Featuretools failed to load plugin module from library featuretools_plugin"
    )
    traceback = "NotImplementedError: plugin not implemented"

    assert message in warning
    assert traceback not in warning
    assert message in debug
    assert traceback in debug
