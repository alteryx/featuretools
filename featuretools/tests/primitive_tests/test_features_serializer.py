import pandas as pd

import featuretools as ft
from featuretools.entityset.deserialize import description_to_entityset
from featuretools.feature_base.features_serializer import FeaturesSerializer
from featuretools.primitives import (
    Count,
    Max,
    MultiplyNumericScalar,
    NMostCommon,
    NumUnique,
)
from featuretools.primitives.utils import serialize_primitive

SCHEMA_VERSION = "9.0.0"


def test_single_feature(es):
    feature = ft.IdentityFeature(es["log"].ww["value"])
    serializer = FeaturesSerializer([feature])

    expected = {
        "ft_version": ft.__version__,
        "schema_version": SCHEMA_VERSION,
        "entityset": es.to_dictionary(),
        "feature_list": [feature.unique_name()],
        "feature_definitions": {feature.unique_name(): feature.to_dictionary()},
        "primitive_definitions": {},
    }

    _compare_feature_dicts(expected, serializer.to_dict())


def test_base_features_in_list(es):
    value = ft.IdentityFeature(es["log"].ww["value"])
    max_feature = ft.AggregationFeature(value, "sessions", Max)
    features = [max_feature, value]
    serializer = FeaturesSerializer(features)

    expected = {
        "ft_version": ft.__version__,
        "schema_version": SCHEMA_VERSION,
        "entityset": es.to_dictionary(),
        "feature_list": [max_feature.unique_name(), value.unique_name()],
        "feature_definitions": {
            max_feature.unique_name(): max_feature.to_dictionary(),
            value.unique_name(): value.to_dictionary(),
        },
    }
    expected["primitive_definitions"] = {"0": serialize_primitive(Max())}
    expected["feature_definitions"][max_feature.unique_name()]["arguments"][
        "primitive"
    ] = "0"

    actual = serializer.to_dict()
    _compare_feature_dicts(expected, actual)


def test_multi_output_features(es):
    product_id = ft.IdentityFeature(es["log"].ww["product_id"])
    threecommon = NMostCommon()
    num_unique = NumUnique()
    tc = ft.Feature(product_id, parent_dataframe_name="sessions", primitive=threecommon)

    features = [tc, product_id]
    for i in range(3):
        features.append(
            ft.Feature(
                tc[i],
                parent_dataframe_name="customers",
                primitive=num_unique,
            )
        )
        features.append(tc[i])

    serializer = FeaturesSerializer(features)

    flist = [feat.unique_name() for feat in features]
    fd = [feat.to_dictionary() for feat in features]
    fdict = dict(zip(flist, fd))

    expected = {
        "ft_version": ft.__version__,
        "schema_version": SCHEMA_VERSION,
        "entityset": es.to_dictionary(),
        "feature_list": flist,
        "feature_definitions": fdict,
    }
    expected["primitive_definitions"] = {
        "0": serialize_primitive(threecommon),
        "1": serialize_primitive(num_unique),
    }

    expected["feature_definitions"][flist[0]]["arguments"]["primitive"] = "0"
    expected["feature_definitions"][flist[2]]["arguments"]["primitive"] = "1"
    expected["feature_definitions"][flist[4]]["arguments"]["primitive"] = "1"
    expected["feature_definitions"][flist[6]]["arguments"]["primitive"] = "1"

    actual = serializer.to_dict()
    _compare_feature_dicts(expected, actual)


def test_base_features_not_in_list(es):
    max_primitive = Max()
    mult_primitive = MultiplyNumericScalar(value=2)
    value = ft.IdentityFeature(es["log"].ww["value"])
    value_x2 = ft.TransformFeature(value, mult_primitive)
    max_feature = ft.AggregationFeature(value_x2, "sessions", max_primitive)
    features = [max_feature]
    serializer = FeaturesSerializer(features)

    expected = {
        "ft_version": ft.__version__,
        "schema_version": SCHEMA_VERSION,
        "entityset": es.to_dictionary(),
        "feature_list": [max_feature.unique_name()],
        "feature_definitions": {
            max_feature.unique_name(): max_feature.to_dictionary(),
            value_x2.unique_name(): value_x2.to_dictionary(),
            value.unique_name(): value.to_dictionary(),
        },
    }
    expected["primitive_definitions"] = {
        "0": serialize_primitive(max_primitive),
        "1": serialize_primitive(mult_primitive),
    }
    expected["feature_definitions"][max_feature.unique_name()]["arguments"][
        "primitive"
    ] = "0"
    expected["feature_definitions"][value_x2.unique_name()]["arguments"][
        "primitive"
    ] = "1"

    actual = serializer.to_dict()
    _compare_feature_dicts(expected, actual)


def test_where_feature_dependency(es):
    max_primitive = Max()
    value = ft.IdentityFeature(es["log"].ww["value"])
    is_purchased = ft.IdentityFeature(es["log"].ww["purchased"])
    max_feature = ft.AggregationFeature(
        value, "sessions", max_primitive, where=is_purchased
    )
    features = [max_feature]
    serializer = FeaturesSerializer(features)

    expected = {
        "ft_version": ft.__version__,
        "schema_version": SCHEMA_VERSION,
        "entityset": es.to_dictionary(),
        "feature_list": [max_feature.unique_name()],
        "feature_definitions": {
            max_feature.unique_name(): max_feature.to_dictionary(),
            value.unique_name(): value.to_dictionary(),
            is_purchased.unique_name(): is_purchased.to_dictionary(),
        },
    }
    expected["primitive_definitions"] = {
        "0": serialize_primitive(max_primitive),
    }
    expected["feature_definitions"][max_feature.unique_name()]["arguments"][
        "primitive"
    ] = "0"

    actual = serializer.to_dict()
    _compare_feature_dicts(expected, actual)


def test_feature_use_previous_pd_timedelta(es):
    value = ft.IdentityFeature(es["log"].ww["id"])
    td = pd.Timedelta(12, "W")
    count_primitive = Count()
    count_feature = ft.AggregationFeature(
        value, "customers", count_primitive, use_previous=td
    )
    features = [count_feature, value]
    serializer = FeaturesSerializer(features)

    expected = {
        "ft_version": ft.__version__,
        "schema_version": SCHEMA_VERSION,
        "entityset": es.to_dictionary(),
        "feature_list": [count_feature.unique_name(), value.unique_name()],
        "feature_definitions": {
            count_feature.unique_name(): count_feature.to_dictionary(),
            value.unique_name(): value.to_dictionary(),
        },
    }
    expected["primitive_definitions"] = {"0": serialize_primitive(count_primitive)}
    expected["feature_definitions"][count_feature.unique_name()]["arguments"][
        "primitive"
    ] = "0"

    actual = serializer.to_dict()
    _compare_feature_dicts(expected, actual)


def test_feature_use_previous_pd_dateoffset(es):
    value = ft.IdentityFeature(es["log"].ww["id"])
    do = pd.DateOffset(months=3)
    count_primitive = Count()
    count_feature = ft.AggregationFeature(
        value, "customers", count_primitive, use_previous=do
    )
    features = [count_feature, value]
    serializer = FeaturesSerializer(features)

    expected = {
        "ft_version": ft.__version__,
        "schema_version": SCHEMA_VERSION,
        "entityset": es.to_dictionary(),
        "feature_list": [count_feature.unique_name(), value.unique_name()],
        "feature_definitions": {
            count_feature.unique_name(): count_feature.to_dictionary(),
            value.unique_name(): value.to_dictionary(),
        },
    }
    expected["primitive_definitions"] = {"0": serialize_primitive(count_primitive)}
    expected["feature_definitions"][count_feature.unique_name()]["arguments"][
        "primitive"
    ] = "0"

    actual = serializer.to_dict()
    _compare_feature_dicts(expected, actual)

    value = ft.IdentityFeature(es["log"].ww["id"])
    do = pd.DateOffset(months=3, days=2, minutes=30)
    count_feature = ft.AggregationFeature(
        value, "customers", count_primitive, use_previous=do
    )
    features = [count_feature, value]
    serializer = FeaturesSerializer(features)

    expected = {
        "ft_version": ft.__version__,
        "schema_version": SCHEMA_VERSION,
        "entityset": es.to_dictionary(),
        "feature_list": [count_feature.unique_name(), value.unique_name()],
        "feature_definitions": {
            count_feature.unique_name(): count_feature.to_dictionary(),
            value.unique_name(): value.to_dictionary(),
        },
    }
    expected["primitive_definitions"] = {"0": serialize_primitive(count_primitive)}
    expected["feature_definitions"][count_feature.unique_name()]["arguments"][
        "primitive"
    ] = "0"
    actual = serializer.to_dict()
    _compare_feature_dicts(expected, actual)


def _compare_feature_dicts(a_dict, b_dict):
    # We can't compare entityset dictionaries because column lists are not
    # guaranteed to be in the same order.
    es_a = description_to_entityset(a_dict.pop("entityset"))
    es_b = description_to_entityset(b_dict.pop("entityset"))
    assert es_a == es_b

    assert a_dict == b_dict
