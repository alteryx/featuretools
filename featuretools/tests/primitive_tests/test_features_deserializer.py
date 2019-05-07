import pytest

from ..testing_utils import make_ecommerce_entityset

import featuretools as ft
from featuretools.feature_base.features_deserializer import (
    FeaturesDeserializer
)
from featuretools.feature_base.features_serializer import SCHEMA_VERSION


@pytest.fixture(scope='module')
def es():
    return make_ecommerce_entityset()


def test_single_feature(es):
    feature = ft.IdentityFeature(es['log']['value'])
    dictionary = {
        'ft_version': ft.__version__,
        'schema_version': SCHEMA_VERSION,
        'entityset': es.to_dictionary(),
        'feature_list': [feature.unique_name()],
        'feature_definitions': {
            feature.unique_name(): feature.to_dictionary()
        }
    }
    deserializer = FeaturesDeserializer(dictionary)

    expected = [feature]
    assert expected == deserializer.to_list()


def test_base_features_in_list(es):
    value = ft.IdentityFeature(es['log']['value'])
    max_feat = ft.AggregationFeature(value, es['sessions'], ft.primitives.Max)
    dictionary = {
        'ft_version': ft.__version__,
        'schema_version': SCHEMA_VERSION,
        'entityset': es.to_dictionary(),
        'feature_list': [max_feat.unique_name(), value.unique_name()],
        'feature_definitions': {
            max_feat.unique_name(): max_feat.to_dictionary(),
            value.unique_name(): value.to_dictionary(),
        }
    }
    deserializer = FeaturesDeserializer(dictionary)

    expected = [max_feat, value]
    assert expected == deserializer.to_list()


def test_base_features_not_in_list(es):
    value = ft.IdentityFeature(es['log']['value'])
    value_x2 = ft.TransformFeature(value,
                                   ft.primitives.MultiplyNumericScalar(value=2))
    max_feat = ft.AggregationFeature(value_x2, es['sessions'], ft.primitives.Max)
    dictionary = {
        'ft_version': ft.__version__,
        'schema_version': SCHEMA_VERSION,
        'entityset': es.to_dictionary(),
        'feature_list': [max_feat.unique_name()],
        'feature_definitions': {
            max_feat.unique_name(): max_feat.to_dictionary(),
            value_x2.unique_name(): value_x2.to_dictionary(),
            value.unique_name(): value.to_dictionary(),
        }
    }
    deserializer = FeaturesDeserializer(dictionary)

    expected = [max_feat]
    assert expected == deserializer.to_list()


def test_later_schema_version(es):
    def test_version(major, minor, patch, raises=True):
        version = '.'.join([str(v) for v in [major, minor, patch]])
        dictionary = {
            'ft_version': ft.__version__,
            'schema_version': version,
            'entityset': es.to_dictionary(),
            'feature_list': [],
            'feature_definitions': {}
        }

        error_text = ('Unable to load features. The schema version of the saved '
                      'features (%s) is greater than the latest supported (%s). '
                      'You may need to upgrade featuretools.'
                      % (version, SCHEMA_VERSION))

        if raises:
            with pytest.raises(RuntimeError) as excinfo:
                FeaturesDeserializer(dictionary)

            assert error_text == str(excinfo.value)
        else:
            FeaturesDeserializer(dictionary)

    major, minor, patch = [int(s) for s in SCHEMA_VERSION.split('.')]
    test_version(major + 1, minor, patch)
    test_version(major, minor + 1, patch)
    test_version(major, minor, patch + 1)
    test_version(major - 1, minor + 1, patch, raises=False)
    test_version(major - 1, minor, patch + 1, raises=False)
    test_version(major, minor - 1, patch + 1, raises=False)
