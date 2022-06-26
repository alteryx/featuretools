import featuretools as ft
from featuretools.primitives.utils import PrimitivesDeserializer
from featuretools import IdentityFeature

def test_relationship_path(es):
    value = IdentityFeature(es["log"].ww["value"])
    assert len(value.relationship_path) == 0


def test_serialization(es):
    value = IdentityFeature(es["log"].ww["value"])

    dictionary = {
        "name": None,
        "column_name": "value",
        "dataframe_name": "log",
    }

    assert dictionary == value.get_arguments()
    assert value == IdentityFeature.from_dictionary(
        dictionary, es, {}, PrimitivesDeserializer
    )
