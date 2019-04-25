import pytest

from featuretools.computational_backends.utils import create_client_and_cluster


def test_create_client_and_cluster():
    match = r'.*workers requested, but only .* workers created'
    with pytest.warns(UserWarning, match=match) as record:
        create_client_and_cluster(1000, 2, {}, 1)
    assert len(record) == 1
