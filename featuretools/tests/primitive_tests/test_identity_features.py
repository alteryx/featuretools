import featuretools as ft
from featuretools.primitives.utils import PrimitivesDeserializer


def test_relationship_path(pd_es):
    value = ft.IdentityFeature(pd_es['log']['value'])
    assert len(value.relationship_path) == 0


def test_serialization(pd_es):
    value = ft.IdentityFeature(pd_es['log']['value'])

    dictionary = {
        'name': None,
        'entity_id': 'log',
        'variable_id': 'value',
    }

    assert dictionary == value.get_arguments()
    assert value == ft.IdentityFeature.from_dictionary(dictionary, pd_es, {},
                                                       PrimitivesDeserializer)
