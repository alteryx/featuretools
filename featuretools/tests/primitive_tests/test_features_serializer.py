import featuretools as ft
from featuretools.entityset.deserialize import description_to_entityset
from featuretools.feature_base.features_serializer import FeaturesSerializer

SCHEMA_VERSION = "1.0.0"


def test_single_feature(es):
    feature = ft.IdentityFeature(es['log']['value'])
    serializer = FeaturesSerializer([feature])

    expected = {
        'ft_version': ft.__version__,
        'schema_version': SCHEMA_VERSION,
        'entityset': es.to_dictionary(),
        'feature_list': [feature.unique_name()],
        'feature_definitions': {
            feature.unique_name(): feature.to_dictionary()
        }
    }

    _compare_feature_dicts(expected, serializer.to_dict())


def test_base_features_in_list(es):
    value = ft.IdentityFeature(es['log']['value'])
    max_feature = ft.AggregationFeature(value, es['sessions'], ft.primitives.Max)
    features = [max_feature, value]
    serializer = FeaturesSerializer(features)

    expected = {
        'ft_version': ft.__version__,
        'schema_version': SCHEMA_VERSION,
        'entityset': es.to_dictionary(),
        'feature_list': [max_feature.unique_name(), value.unique_name()],
        'feature_definitions': {
            max_feature.unique_name(): max_feature.to_dictionary(),
            value.unique_name(): value.to_dictionary(),
        }
    }

    _compare_feature_dicts(expected, serializer.to_dict())


def test_base_features_not_in_list(es):
    value = ft.IdentityFeature(es['log']['value'])
    value_x2 = ft.TransformFeature(value,
                                   ft.primitives.MultiplyNumericScalar(value=2))
    max_feature = ft.AggregationFeature(value_x2, es['sessions'], ft.primitives.Max)
    features = [max_feature]
    serializer = FeaturesSerializer(features)

    expected = {
        'ft_version': ft.__version__,
        'schema_version': SCHEMA_VERSION,
        'entityset': es.to_dictionary(),
        'feature_list': [max_feature.unique_name()],
        'feature_definitions': {
            max_feature.unique_name(): max_feature.to_dictionary(),
            value_x2.unique_name(): value_x2.to_dictionary(),
            value.unique_name(): value.to_dictionary(),
        }
    }

    _compare_feature_dicts(expected, serializer.to_dict())


def test_where_feature_dependency(es):
    value = ft.IdentityFeature(es['log']['value'])
    is_purchased = ft.IdentityFeature(es['log']['purchased'])
    max_feature = ft.AggregationFeature(value, es['sessions'], ft.primitives.Max,
                                        where=is_purchased)
    features = [max_feature]
    serializer = FeaturesSerializer(features)

    expected = {
        'ft_version': ft.__version__,
        'schema_version': SCHEMA_VERSION,
        'entityset': es.to_dictionary(),
        'feature_list': [max_feature.unique_name()],
        'feature_definitions': {
            max_feature.unique_name(): max_feature.to_dictionary(),
            value.unique_name(): value.to_dictionary(),
            is_purchased.unique_name(): is_purchased.to_dictionary(),
        }
    }

    _compare_feature_dicts(expected, serializer.to_dict())


def _compare_feature_dicts(a, b):
    # We can't compare entityset dictionaries because variable lists are not
    # guaranteed to be in the same order.
    es_a = description_to_entityset(a.pop('entityset'))
    es_b = description_to_entityset(b.pop('entityset'))
    assert es_a == es_b

    assert a == b
