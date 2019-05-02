import pytest

from ..testing_utils import make_ecommerce_entityset

import featuretools as ft


@pytest.fixture
def es():
    return make_ecommerce_entityset()


def test_serialization(es):
    value = ft.IdentityFeature(es['log']['value'])

    dictionary = {
        'entity_id': 'log',
        'variable_id': 'value',
    }

    assert dictionary == value.get_arguments()
    assert value == ft.IdentityFeature.from_dictionary(dictionary, es, {})
