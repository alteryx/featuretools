import pytest

from ..testing_utils import make_ecommerce_entityset

import featuretools as ft
from featuretools.entityset.deserialize import description_to_entityset
from featuretools.feature_base.features_serializer import FeaturesSerializer

SCHEMA_VERSION = "1.0.0"


@pytest.fixture(scope='module')
def es():
    return make_ecommerce_entityset()


def test_single_feature(es):
    feature = ft.IdentityFeature(es['log']['value'])
    serializer = FeaturesSerializer([feature])

    expected = {
        'ft_version': ft.__version__,
        'schema_version': SCHEMA_VERSION,
        'entityset': es.to_dictionary(),
        'feature_list': [feature.unique_name()],
        'feature_definitions': {
            feature.unique_name(): _feature_dict(feature)
        }
    }

    _compare_feature_dicts(expected, serializer.to_dict())


def test_base_features_in_list(es):
    value = ft.IdentityFeature(es['log']['value'])
    max = ft.AggregationFeature(value, es['sessions'], ft.primitives.Max)
    features = [max, value]
    serializer = FeaturesSerializer(features)

    expected = {
        'ft_version': ft.__version__,
        'schema_version': SCHEMA_VERSION,
        'entityset': es.to_dictionary(),
        'feature_list': [max.unique_name(), value.unique_name()],
        'feature_definitions': {
            max.unique_name(): _feature_dict(max),
            value.unique_name(): _feature_dict(value),
        }
    }

    _compare_feature_dicts(expected, serializer.to_dict())


def test_base_features_not_in_list(es):
    value = ft.IdentityFeature(es['log']['value'])
    value_x2 = ft.TransformFeature(value,
                                   ft.primitives.MultiplyNumericScalar(value=2))
    max = ft.AggregationFeature(value_x2, es['sessions'], ft.primitives.Max)
    features = [max]
    serializer = FeaturesSerializer(features)

    expected = {
        'ft_version': ft.__version__,
        'schema_version': SCHEMA_VERSION,
        'entityset': es.to_dictionary(),
        'feature_list': [max.unique_name()],
        'feature_definitions': {
            max.unique_name(): _feature_dict(max),
            value_x2.unique_name(): _feature_dict(value_x2),
            value.unique_name(): _feature_dict(value),
        }
    }

    _compare_feature_dicts(expected, serializer.to_dict())


def test_where_feature_dependency(es):
    value = ft.IdentityFeature(es['log']['value'])
    is_purchased = ft.IdentityFeature(es['log']['purchased'])
    max = ft.AggregationFeature(value, es['sessions'], ft.primitives.Max,
                                where=is_purchased)
    features = [max]
    serializer = FeaturesSerializer(features)

    expected = {
        'ft_version': ft.__version__,
        'schema_version': SCHEMA_VERSION,
        'entityset': es.to_dictionary(),
        'feature_list': [max.unique_name()],
        'feature_definitions': {
            max.unique_name(): _feature_dict(max),
            value.unique_name(): _feature_dict(value),
            is_purchased.unique_name(): _feature_dict(is_purchased),
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


def _feature_dict(feature):
    return {
        'type': type(feature).__name__,
        'dependencies': [dep.unique_name() for dep in feature.get_dependencies()],
        'arguments': feature.get_arguments()
    }
