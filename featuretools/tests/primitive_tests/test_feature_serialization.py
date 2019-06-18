import os

from pympler.asizeof import asizeof

import featuretools as ft
from featuretools.feature_base.features_deserializer import (
    FeaturesDeserializer
)
from featuretools.feature_base.features_serializer import FeaturesSerializer
from featuretools.primitives import CumSum, make_agg_primitive
from featuretools.variable_types import Numeric


def pickle_features_test_helper(es_size, features_original):
    dir_path = os.path.dirname(os.path.realpath(__file__))
    filepath = os.path.join(dir_path, 'test_feature')

    ft.save_features(features_original, filepath)
    features_deserializedA = ft.load_features(filepath)
    assert os.path.getsize(filepath) < es_size
    os.remove(filepath)

    with open(filepath, "w") as f:
        ft.save_features(features_original, f)
    features_deserializedB = ft.load_features(open(filepath))
    assert os.path.getsize(filepath) < es_size
    os.remove(filepath)

    features = ft.save_features(features_original)
    features_deserializedC = ft.load_features(features)
    assert asizeof(features) < es_size

    features_deserialized_options = [features_deserializedA, features_deserializedB, features_deserializedC]
    for features_deserialized in features_deserialized_options:
        for feat_1, feat_2 in zip(features_original, features_deserialized):
            assert feat_1.unique_name() == feat_2.unique_name()
            assert feat_1.entityset == feat_2.entityset


def test_pickle_features(es):
    features_original = ft.dfs(target_entity='sessions', entityset=es, features_only=True)
    pickle_features_test_helper(asizeof(es), features_original)


def test_pickle_features_with_custom_primitive(es):
    NewMax = make_agg_primitive(
        lambda x: max(x),
        name="NewMax",
        input_types=[Numeric],
        return_type=Numeric,
        description="Calculate means ignoring nan values")

    features_original = ft.dfs(target_entity='sessions', entityset=es,
                               agg_primitives=["Last", "Mean", NewMax], features_only=True)

    assert any([isinstance(feat.primitive, NewMax) for feat in features_original])
    pickle_features_test_helper(asizeof(es), features_original)


def test_serialized_renamed_features(es):
    def serialize_name_unchanged(original):
        renamed = original.rename('MyFeature')
        assert renamed.get_name() == 'MyFeature'

        serializer = FeaturesSerializer([renamed])
        serialized = serializer.to_dict()

        deserializer = FeaturesDeserializer(serialized)
        deserialized = deserializer.to_list()[0]
        assert deserialized.get_name() == 'MyFeature'

    identity_original = ft.IdentityFeature(es['log']['value'])
    assert identity_original.get_name() == 'value'

    value = ft.IdentityFeature(es['log']['value'])

    primitive = ft.primitives.Max()
    agg_original = ft.AggregationFeature(value, es['customers'], primitive)
    assert agg_original.get_name() == 'MAX(log.value)'

    direct_original = ft.DirectFeature(es['customers']['age'], es['sessions'])
    assert direct_original.get_name() == 'customers.age'

    primitive = ft.primitives.MultiplyNumericScalar(value=2)
    transform_original = ft.TransformFeature(value, primitive)
    assert transform_original.get_name() == 'value * 2'

    zipcode = ft.IdentityFeature(es['log']['zipcode'])
    primitive = CumSum()
    groupby_original = ft.feature_base.GroupByTransformFeature(value, primitive, zipcode)
    assert groupby_original.get_name() == 'CUM_SUM(value) by zipcode'

    feature_type_list = [identity_original, agg_original, direct_original, transform_original, groupby_original]

    for feature_type in feature_type_list:
        serialize_name_unchanged(feature_type)
