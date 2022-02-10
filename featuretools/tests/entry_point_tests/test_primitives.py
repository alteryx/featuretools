from featuretools.tests.entry_point_tests.utils import (
    import_featuretools,
    install_featuretools_primitives,
    uninstall_featuretools_primitives
)


def test_entry_point():
    install_featuretools_primitives()
    subprocess = import_featuretools('debug')
    uninstall_featuretools_primitives()
    log = subprocess.stdout.decode()

    invalid_primitive = 'Featuretools failed to load "invalid" primitives from "featuretools_primitives.invalid_primitive". '
    invalid_primitive += 'For a full stack trace, set logging to debug.'
    assert invalid_primitive in log

    existing_primitive = 'Ignoring primitive "Sum" from "featuretools_primitives.existing_primitive" because it '
    existing_primitive += 'already exists in "featuretools.primitives.standard.aggregation_primitives"'
    assert existing_primitive in log
