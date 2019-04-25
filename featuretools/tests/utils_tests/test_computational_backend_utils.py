import warnings

from featuretools.computational_backends.utils import create_client_and_cluster


def test_create_client_and_cluster():
    with warnings.catch_warnings(record=True) as w:
        # Cause all warnings to always be triggered.
        warnings.simplefilter("always")

        create_client_and_cluster(10, 2, {}, 1)
        assert len(w) == 1
