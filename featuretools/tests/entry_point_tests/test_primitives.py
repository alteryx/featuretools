from featuretools.tests.entry_point_tests.utils import (
    _import_featuretools,
    _install_featuretools_primitives,
    _python,
    _uninstall_featuretools_primitives,
)


def test_entry_point():
    _install_featuretools_primitives()
    featuretools_log = _import_featuretools("debug").stdout.decode()
    new_primitive = _python("-c", "from featuretools.primitives import NewPrimitive")
    _uninstall_featuretools_primitives()
    assert new_primitive.returncode == 0

    invalid_primitive = 'Featuretools failed to load "invalid" primitives from "featuretools_primitives.invalid_primitive". '
    invalid_primitive += "For a full stack trace, set logging to debug."
    assert invalid_primitive in featuretools_log

    existing_primitive = 'While loading primitives via "existing" entry point, '
    existing_primitive += 'ignored primitive "Sum" from "featuretools_primitives.existing_primitive" because a primitive '
    existing_primitive += 'with that name already exists in "featuretools.primitives.standard.aggregation.sum_primitive"'
    assert existing_primitive in featuretools_log
