import pandas as pd

import featuretools as ft
from featuretools.entityset.deserialize import description_to_entityset
from featuretools.feature_base.features_serializer import FeaturesSerializer

SCHEMA_VERSION = "5.0.0"


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


def test_multi_output_features(es):
    value = ft.IdentityFeature(es['log']['product_id'])
    threecommon = ft.primitives.NMostCommon()
    tc = ft.Feature(es['log']['product_id'], parent_entity=es["sessions"], primitive=threecommon)

    features = [tc, value]
    for i in range(3):
        features.append(ft.Feature(tc[i],
                                   parent_entity=es['customers'],
                                   primitive=ft.primitives.NumUnique))
        features.append(tc[i])

    serializer = FeaturesSerializer(features)

    flist = [feat.unique_name() for feat in features]
    fd = [feat.to_dictionary() for feat in features]
    fdict = dict(zip(flist, fd))

    expected = {
        'ft_version': ft.__version__,
        'schema_version': SCHEMA_VERSION,
        'entityset': es.to_dictionary(),
        'feature_list': flist,
        'feature_definitions': fdict
    }
    actual = serializer.to_dict()

    _compare_feature_dicts(expected, actual)


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


def test_feature_use_previous_pd_timedelta(es):
    value = ft.IdentityFeature(es['log']['id'])
    td = pd.Timedelta(12, "W")
    count_feature = ft.AggregationFeature(value, es['customers'], ft.primitives.Count, use_previous=td)
    features = [count_feature, value]
    serializer = FeaturesSerializer(features)

    expected = {
        'ft_version': ft.__version__,
        'schema_version': SCHEMA_VERSION,
        'entityset': es.to_dictionary(),
        'feature_list': [count_feature.unique_name(), value.unique_name()],
        'feature_definitions': {
            count_feature.unique_name(): count_feature.to_dictionary(),
            value.unique_name(): value.to_dictionary(),
        }
    }

    _compare_feature_dicts(expected, serializer.to_dict())


def test_feature_use_previous_pd_dateoffset(es):
    value = ft.IdentityFeature(es['log']['id'])
    do = pd.DateOffset(months=3)
    count_feature = ft.AggregationFeature(value, es['customers'], ft.primitives.Count, use_previous=do)
    features = [count_feature, value]
    serializer = FeaturesSerializer(features)

    expected = {
        'ft_version': ft.__version__,
        'schema_version': SCHEMA_VERSION,
        'entityset': es.to_dictionary(),
        'feature_list': [count_feature.unique_name(), value.unique_name()],
        'feature_definitions': {
            count_feature.unique_name(): count_feature.to_dictionary(),
            value.unique_name(): value.to_dictionary(),
        }
    }

    _compare_feature_dicts(expected, serializer.to_dict())

    value = ft.IdentityFeature(es['log']['id'])
    do = pd.DateOffset(months=3, days=2, minutes=30)
    count_feature = ft.AggregationFeature(value, es['customers'], ft.primitives.Count, use_previous=do)
    features = [count_feature, value]
    serializer = FeaturesSerializer(features)

    expected = {
        'ft_version': ft.__version__,
        'schema_version': SCHEMA_VERSION,
        'entityset': es.to_dictionary(),
        'feature_list': [count_feature.unique_name(), value.unique_name()],
        'feature_definitions': {
            count_feature.unique_name(): count_feature.to_dictionary(),
            value.unique_name(): value.to_dictionary(),
        }
    }

    _compare_feature_dicts(expected, serializer.to_dict())


def _compare_feature_dicts(a_dict, b_dict):
    # We can't compare entityset dictionaries because variable lists are not
    # guaranteed to be in the same order.
    es_a = description_to_entityset(a_dict.pop('entityset'))
    es_b = description_to_entityset(b_dict.pop('entityset'))
    assert es_a == es_b

    assert a_dict == b_dict
