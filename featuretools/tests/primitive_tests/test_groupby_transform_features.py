import featuretools as ft
from featuretools.feature_base.features_deserializer import (
    FeaturesDeserializer
)
from featuretools.feature_base.features_serializer import FeaturesSerializer
from featuretools.primitives import CumSum


def test_rename_serialization(es):
    value = ft.IdentityFeature(es['log']['value'])
    zipcode = ft.IdentityFeature(es['log']['zipcode'])
    primitive = CumSum()
    groupby = ft.feature_base.GroupByTransformFeature(value, primitive, zipcode)
    assert groupby.get_name() == 'CUM_SUM(value) by zipcode'

    renamed = groupby.rename('MyFeature')
    assert renamed.get_name() == 'MyFeature'

    serializer = FeaturesSerializer([renamed])
    serialized = serializer.to_dict()

    deserializer = FeaturesDeserializer(serialized)
    deserialized = deserializer.to_list()[0]
    assert deserialized.get_name() == 'MyFeature'
