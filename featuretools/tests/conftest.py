import pytest

from featuretools.tests.testing_utils import make_ecommerce_entityset


@pytest.fixture
def es():
    return make_ecommerce_entityset()


@pytest.fixture
def int_es():
    return make_ecommerce_entityset(with_integer_time_index=True)
