from featuretools.tests.entry_point_tests.utils import (
    import_featuretools,
    install_featuretools_primitives,
    uninstall_featuretools_primitives
)


def test_entry_point():
    install_featuretools_primitives()
    process = import_featuretools('warning')
    uninstall_featuretools_primitives()

    expected = 'Ignoring primitive "Sum" from "featuretools_primitives" because it '
    expected += 'already exists in "featuretools.primitives.standard.aggregation_primitives"'
    actual = process.stdout.decode()
    assert expected in actual
