import featuretools as ft
from featuretools.primitives.utils import PrimitivesDeserializer


def test_relationship_path(es):
    value = ft.IdentityFeature(es['log'].ww['value'])
    assert len(value.relationship_path) == 0


def test_serialization(es):
    value = ft.IdentityFeature(es['log'].ww['value'])

    dictionary = {
        'name': None,
        'column_name': 'value',
        'dataframe_name': 'log',
    }

    assert dictionary == value.get_arguments()
    assert value == ft.IdentityFeature.from_dictionary(dictionary, es, {},
                                                       PrimitivesDeserializer)
