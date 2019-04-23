# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import pytest

from ..testing_utils import make_ecommerce_entityset

import featuretools as ft
from featuretools.computational_backends import PandasBackend
from featuretools.primitives import (
    Absolute,
    AddNumeric,
    AddNumericScalar,
    Count,
    Day,
    Diff,
    DivideByFeature,
    DivideNumeric,
    DivideNumericScalar,
    Equal,
    EqualScalar,
    GreaterThanEqualToScalar,
    GreaterThanScalar,
    Haversine,
    Hour,
    IsIn,
    IsNull,
    Latitude,
    LessThanEqualToScalar,
    LessThanScalar,
    Longitude,
    Minute,
    Mode,
    Month,
    MultiplyNumeric,
    MultiplyNumericScalar,
    NMostCommon,
    Not,
    NotEqual,
    NotEqualScalar,
    NumCharacters,
    NumWords,
    Percentile,
    ScalarSubtractNumericFeature,
    Second,
    SubtractNumeric,
    SubtractNumericScalar,
    Sum,
    TransformPrimitive,
    Year,
    get_transform_primitives
)
from featuretools.primitives.base import make_trans_primitive
from featuretools.synthesis.deep_feature_synthesis import match
from featuretools.variable_types import Boolean, Datetime, Numeric, Variable


# some tests change the entityset values, so we have to create it fresh
# for each test (rather than setting scope='module')
@pytest.fixture
def es():
    return make_ecommerce_entityset()


@pytest.fixture(scope='module')
def int_es():
    return make_ecommerce_entityset(with_integer_time_index=True)


def test_init_and_name(es):
    log = es['log']
    rating = ft.Feature(es["products"]["rating"], es["log"])
    features = [ft.Feature(v) for v in log.variables] +\
        [ft.Feature(rating, primitive=GreaterThanScalar(2.5))]
    # Add Timedelta feature
    # features.append(pd.Timestamp.now() - ft.Feature(log['datetime']))
    for transform_prim in get_transform_primitives().values():

        # skip automated testing if a few special cases
        if transform_prim in [NotEqual, Equal]:
            continue

        # use the input_types matching function from DFS
        input_types = transform_prim.input_types
        if type(input_types[0]) == list:
            matching_inputs = match(input_types[0], features)
        else:
            matching_inputs = match(input_types, features)
        if len(matching_inputs) == 0:
            raise Exception(
                "Transform Primitive %s not tested" % transform_prim.name)
        for s in matching_inputs:
            instance = ft.Feature(s, primitive=transform_prim)

            # try to get name and calculate
            instance.get_name()
            ft.calculate_feature_matrix([instance], entityset=es).head(5)


def test_make_trans_feat(es):
    f = ft.Feature(es['log']['datetime'], primitive=Hour)

    pandas_backend = PandasBackend(es, [f])
    df = pandas_backend.calculate_all_features(instance_ids=[0],
                                               time_last=None)
    v = df[f.get_name()][0]
    assert v == 10


def test_diff(es):
    value = ft.Feature(es['log']['value'])
    customer_id_feat = ft.Feature(es['sessions']['customer_id'], entity=es['log'])
    diff1 = ft.Feature([value, es['log']['session_id']], primitive=Diff)
    diff2 = ft.Feature([value, customer_id_feat], primitive=Diff)

    pandas_backend = PandasBackend(es, [diff1, diff2])
    df = pandas_backend.calculate_all_features(instance_ids=range(15),
                                               time_last=None)

    val1 = df[diff1.get_name()].values.tolist()
    val2 = df[diff2.get_name()].values.tolist()
    correct_vals1 = [
        np.nan, 5, 5, 5, 5, np.nan, 1, 1, 1, np.nan, np.nan, 5, np.nan, 7, 7
    ]
    correct_vals2 = [np.nan, 5, 5, 5, 5, -20, 1, 1, 1, -3, np.nan, 5, -5, 7, 7]
    for i, v in enumerate(val1):
        v1 = val1[i]
        if np.isnan(v1):
            assert (np.isnan(correct_vals1[i]))
        else:
            assert v1 == correct_vals1[i]
        v2 = val2[i]
        if np.isnan(v2):
            assert (np.isnan(correct_vals2[i]))
        else:
            assert v2 == correct_vals2[i]


def test_diff_single_value(es):
    diff = ft.Feature([es['stores']['num_square_feet'], es['stores'][u'région_id']], primitive=Diff)
    pandas_backend = PandasBackend(es, [diff])
    df = pandas_backend.calculate_all_features(instance_ids=[5],
                                               time_last=None)
    assert df.shape[0] == 1
    assert df[diff.get_name()].dropna().shape[0] == 0


def test_compare_of_identity(es):
    to_test = [(EqualScalar, [False, False, True, False]),
               (NotEqualScalar, [True, True, False, True]),
               (LessThanScalar, [True, True, False, False]),
               (LessThanEqualToScalar, [True, True, True, False]),
               (GreaterThanScalar, [False, False, False, True]),
               (GreaterThanEqualToScalar, [False, False, True, True])]

    features = []
    for test in to_test:
        features.append(ft.Feature(es['log']['value'], primitive=test[0](10)))

    df = ft.calculate_feature_matrix(entityset=es, features=features, instance_ids=[0, 1, 2, 3])

    for i, test in enumerate(to_test):
        v = df[features[i].get_name()].values.tolist()
        assert v == test[1]


def test_compare_of_direct(es):
    log_rating = ft.Feature(es['products']['rating'], entity=es['log'])
    to_test = [(EqualScalar, [False, False, False, False]),
               (NotEqualScalar, [True, True, True, True]),
               (LessThanScalar, [False, False, False, True]),
               (LessThanEqualToScalar, [False, False, False, True]),
               (GreaterThanScalar, [True, True, True, False]),
               (GreaterThanEqualToScalar, [True, True, True, False])]

    features = []
    for test in to_test:
        features.append(ft.Feature(log_rating, primitive=test[0](4.5)))

    df = ft.calculate_feature_matrix(entityset=es, features=features, instance_ids=[0, 1, 2, 3])

    for i, test in enumerate(to_test):
        v = df[features[i].get_name()].values.tolist()
        assert v == test[1]


def test_compare_of_transform(es):
    day = ft.Feature(es['log']['datetime'], primitive=Day)
    to_test = [(EqualScalar, [False, True]),
               (NotEqualScalar, [True, False]),
               (LessThanScalar, [True, False]),
               (LessThanEqualToScalar, [True, True]),
               (GreaterThanScalar, [False, False]),
               (GreaterThanEqualToScalar, [False, True])]

    features = []
    for test in to_test:
        features.append(ft.Feature(day, primitive=test[0](10)))

    df = ft.calculate_feature_matrix(entityset=es, features=features, instance_ids=[0, 14])

    for i, test in enumerate(to_test):
        v = df[features[i].get_name()].values.tolist()
        assert v == test[1]


def test_compare_of_agg(es):
    count_logs = ft.Feature(es['log']['id'], parent_entity=es['sessions'], primitive=Count)

    to_test = [(EqualScalar, [False, False, False, True]),
               (NotEqualScalar, [True, True, True, False]),
               (LessThanScalar, [False, False, True, False]),
               (LessThanEqualToScalar, [False, False, True, True]),
               (GreaterThanScalar, [True, True, False, False]),
               (GreaterThanEqualToScalar, [True, True, False, True])]

    features = []
    for test in to_test:
        features.append(ft.Feature(count_logs, primitive=test[0](2)))

    df = ft.calculate_feature_matrix(entityset=es, features=features, instance_ids=[0, 1, 2, 3])

    for i, test in enumerate(to_test):
        v = df[features[i].get_name()].values.tolist()
        assert v == test[1]


def test_compare_all_nans(es):
    nan_feat = ft.Feature(es['log']['product_id'], parent_entity=es['sessions'], primitive=Mode)
    compare = nan_feat == 'brown bag'
    # before all data
    time_last = pd.Timestamp('1/1/1993')

    df = ft.calculate_feature_matrix(entityset=es, features=[nan_feat, compare], instance_ids=[0, 1, 2], cutoff_time=time_last)
    assert df[nan_feat.get_name()].dropna().shape[0] == 0
    assert not df[compare.get_name()].any()


def test_arithmetic_of_val(es):
    to_test = [(AddNumericScalar, [2.0, 7.0, 12.0, 17.0]),
               (SubtractNumericScalar, [-2.0, 3.0, 8.0, 13.0]),
               (ScalarSubtractNumericFeature, [2.0, -3.0, -8.0, -13.0]),
               (MultiplyNumericScalar, [0, 10, 20, 30]),
               (DivideNumericScalar, [0, 2.5, 5, 7.5]),
               (DivideByFeature, [np.inf, 0.4, 0.2, 2 / 15.0])]

    features = []
    for test in to_test:
        features.append(ft.Feature(es['log']['value'], primitive=test[0](2)))

    features.append(ft.Feature(es['log']['value']) / 0)

    df = ft.calculate_feature_matrix(entityset=es, features=features, instance_ids=[0, 1, 2, 3])

    for f, test in zip(features, to_test):
        v = df[f.get_name()].values.tolist()
        assert v == test[1]

    test = [np.nan, np.inf, np.inf, np.inf]
    v = df[features[-1].get_name()].values.tolist()
    assert (np.isnan(v[0]))
    assert v[1:] == test[1:]


def test_arithmetic_two_vals_fails(es):
    error_text = "Not a feature"
    with pytest.raises(Exception, match=error_text):
        ft.Feature([2, 2], primitive=AddNumeric)


def test_arithmetic_of_identity(es):
    logs = es['log']

    to_test = [(AddNumeric, [0., 7., 14., 21.]),
               (SubtractNumeric, [0, 3, 6, 9]),
               (MultiplyNumeric, [0, 10, 40, 90]),
               (DivideNumeric, [np.nan, 2.5, 2.5, 2.5])]

    features = []
    for test in to_test:
        features.append(ft.Feature([logs['value'], logs['value_2']], primitive=test[0]))

    df = ft.calculate_feature_matrix(entityset=es, features=features, instance_ids=[0, 1, 2, 3])

    for i, test in enumerate(to_test[:-1]):
        v = df[features[i].get_name()].values.tolist()
        assert v == test[1]
    i, test = 3, to_test[-1]
    v = df[features[i].get_name()].values.tolist()
    assert (np.isnan(v[0]))
    assert v[1:] == test[1][1:]


def test_arithmetic_of_direct(es):
    rating = es['products']['rating']
    log_rating = ft.Feature(rating, entity=es['log'])
    customer_age = es['customers']['age']
    session_age = ft.Feature(customer_age, entity=es['sessions'])
    log_age = ft.Feature(session_age, entity=es['log'])

    to_test = [(AddNumeric, [38, 37, 37.5, 37.5]),
               (SubtractNumeric, [28, 29, 28.5, 28.5]),
               (MultiplyNumeric, [165, 132, 148.5, 148.5]),
               (DivideNumeric, [6.6, 8.25, 22. / 3, 22. / 3])]

    features = []
    for test in to_test:
        features.append(ft.Feature([log_age, log_rating], primitive=test[0]))

    df = ft.calculate_feature_matrix(entityset=es, features=features, instance_ids=[0, 3, 5, 7])
    for i, test in enumerate(to_test):
        v = df[features[i].get_name()].values.tolist()
        assert v == test[1]


# P TODO: rewrite this  test
def test_arithmetic_of_transform(es):
    diff1 = ft.Feature([es['log']['value'], es['log']['product_id']], primitive=Diff)
    diff2 = ft.Feature([es['log']['value_2'], es['log']['product_id']], primitive=Diff)

    to_test = [(AddNumeric, [np.nan, 14., -7., 3.]),
               (SubtractNumeric, [np.nan, 6., -3., 1.]),
               (MultiplyNumeric, [np.nan, 40., 10., 2.]),
               (DivideNumeric, [np.nan, 2.5, 2.5, 2.])]

    features = []
    for test in to_test:
        features.append(ft.Feature([diff1, diff2], primitive=test[0]()))

    pandas_backend = PandasBackend(es, features)
    df = pandas_backend.calculate_all_features(instance_ids=[0, 2, 11, 13],
                                               time_last=None)
    for i, test in enumerate(to_test):
        v = df[features[i].get_name()].values.tolist()
        assert np.isnan(v.pop(0))
        assert np.isnan(test[1].pop(0))
        assert v == test[1]


def test_not_feature(es):
    not_feat = ft.Feature(es['customers']['loves_ice_cream'], primitive=Not)
    features = [not_feat]
    df = ft.calculate_feature_matrix(entityset=es, features=features, instance_ids=[0, 1])
    v = df[not_feat.get_name()].values
    assert not v[0]
    assert v[1]


def test_arithmetic_of_agg(es):
    customer_id_feat = es['customers']['id']
    store_id_feat = es['stores']['id']
    count_customer = ft.Feature(customer_id_feat, parent_entity=es[u'régions'], primitive=Count)
    count_stores = ft.Feature(store_id_feat, parent_entity=es[u'régions'], primitive=Count)
    to_test = [(AddNumeric, [6, 2]),
               (SubtractNumeric, [0, -2]),
               (MultiplyNumeric, [9, 0]),
               (DivideNumeric, [1, 0])]

    features = []
    for test in to_test:
        features.append(ft.Feature([count_customer, count_stores], primitive=test[0]()))

    ids = ['United States', 'Mexico']
    df = ft.calculate_feature_matrix(entityset=es, features=features,
                                     instance_ids=ids)
    df = df.loc[ids]

    for i, test in enumerate(to_test):
        v = df[features[i].get_name()].values.tolist()
        assert v == test[1]


# TODO latlong is a string in entityset. Asserts in test_latlong fail
# def latlong_unstringify(latlong):
#     lat = float(latlong.split(", ")[0].replace("(", ""))
#     lon = float(latlong.split(", ")[1].replace(")", ""))
#     return (lat, lon)


def test_latlong(es):
    log_latlong_feat = es['log']['latlong']
    latitude = ft.Feature(log_latlong_feat, primitive=Latitude)
    longitude = ft.Feature(log_latlong_feat, primitive=Longitude)
    features = [latitude, longitude]
    df = ft.calculate_feature_matrix(entityset=es, features=features, instance_ids=range(15))
    latvalues = df[latitude.get_name()].values
    lonvalues = df[longitude.get_name()].values
    assert len(latvalues) == 15
    assert len(lonvalues) == 15
    real_lats = [0, 5, 10, 15, 20, 0, 1, 2, 3, 0, 0, 5, 0, 7, 14]
    real_lons = [0, 2, 4, 6, 8, 0, 1, 2, 3, 0, 0, 2, 0, 3, 6]
    for i, v, in enumerate(real_lats):
        assert v == latvalues[i]
    for i, v, in enumerate(real_lons):
        assert v == lonvalues[i]


def test_haversine(es):
    log_latlong_feat = es['log']['latlong']
    log_latlong_feat2 = es['log']['latlong2']
    haversine = ft.Feature([log_latlong_feat, log_latlong_feat2],
                           primitive=Haversine)
    features = [haversine]

    df = ft.calculate_feature_matrix(entityset=es, features=features,
                                     instance_ids=range(15))
    values = df[haversine.get_name()].values
    real = [0, 525.318462, 1045.32190304, 1554.56176802, 2047.3294327, 0,
            138.16578931, 276.20524822, 413.99185444, 0, 0, 525.318462, 0,
            741.57941183, 1467.52760175]
    assert len(values) == 15
    assert np.allclose(values, real, atol=0.0001)

    haversine = ft.Feature([log_latlong_feat, log_latlong_feat2],
                           primitive=Haversine(unit='kilometers'))
    features = [haversine]
    df = ft.calculate_feature_matrix(entityset=es, features=features,
                                     instance_ids=range(15))
    values = df[haversine.get_name()].values
    real_km = [0, 845.41812212, 1682.2825471, 2501.82467535, 3294.85736668,
               0, 222.35628593, 444.50926278, 666.25531268, 0, 0,
               845.41812212, 0, 1193.45638714, 2361.75676089]
    assert len(values) == 15
    assert np.allclose(values, real_km, atol=0.0001)
    error_text = "Invalid unit inches provided. Must be one of"
    with pytest.raises(ValueError, match=error_text):
        Haversine(unit='inches')


def test_text_primitives(es):
    words = ft.Feature(es['log']['comments'], primitive=NumWords)
    chars = ft.Feature(es['log']['comments'], primitive=NumCharacters)

    features = [words, chars]

    df = ft.calculate_feature_matrix(entityset=es, features=features, instance_ids=range(15))

    word_counts = [514, 3, 3, 644, 1268, 1269, 177, 172, 79,
                   240, 1239, 3, 3, 3, 3]
    char_counts = [3392, 10, 10, 4116, 7961, 7580, 992, 957,
                   437, 1325, 6322, 10, 10, 10, 10]
    word_values = df[words.get_name()].values
    char_values = df[chars.get_name()].values
    assert len(word_values) == 15
    for i, v in enumerate(word_values):
        assert v == word_counts[i]
    for i, v in enumerate(char_values):
        assert v == char_counts[i]


def test_isin_feat(es):
    isin = ft.Feature(es['log']['product_id'], primitive=IsIn(list_of_outputs=["toothpaste", "coke zero"]))
    features = [isin]
    df = ft.calculate_feature_matrix(entityset=es, features=features, instance_ids=range(8))
    true = [True, True, True, False, False, True, True, True]
    v = df[isin.get_name()].values.tolist()
    assert true == v


def test_isin_feat_other_syntax(es):
    isin = ft.Feature(es['log']['product_id']).isin(["toothpaste", "coke zero"])
    features = [isin]
    df = ft.calculate_feature_matrix(entityset=es, features=features, instance_ids=range(8))
    true = [True, True, True, False, False, True, True, True]
    v = df[isin.get_name()].values.tolist()
    assert true == v


def test_isin_feat_other_syntax_int(es):
    isin = ft.Feature(es['log']['value']).isin([5, 10])
    features = [isin]
    df = ft.calculate_feature_matrix(entityset=es, features=features, instance_ids=range(8))
    true = [False, True, True, False, False, False, False, False]
    v = df[isin.get_name()].values.tolist()
    assert true == v


def test_isin_feat_custom(es):
    def pd_is_in(array, list_of_outputs=None):
        if list_of_outputs is None:
            list_of_outputs = []
        return pd.Series(array).isin(list_of_outputs)

    def isin_generate_name(self, base_feature_names):
        return u"%s.isin(%s)" % (base_feature_names[0],
                                 str(self.kwargs['list_of_outputs']))

    IsIn = make_trans_primitive(
        pd_is_in,
        [Variable],
        Boolean,
        name="is_in",
        description="For each value of the base feature, checks whether it is "
        "in a list that is provided.",
        cls_attributes={"generate_name": isin_generate_name})

    isin = ft.Feature(es['log']['product_id'], primitive=IsIn(list_of_outputs=["toothpaste", "coke zero"]))
    features = [isin]
    df = ft.calculate_feature_matrix(entityset=es, features=features, instance_ids=range(8))
    true = [True, True, True, False, False, True, True, True]
    v = df[isin.get_name()].values.tolist()
    assert true == v

    isin = ft.Feature(es['log']['product_id']).isin(["toothpaste", "coke zero"])
    features = [isin]
    df = ft.calculate_feature_matrix(entityset=es, features=features, instance_ids=range(8))
    true = [True, True, True, False, False, True, True, True]
    v = df[isin.get_name()].values.tolist()
    assert true == v

    isin = ft.Feature(es['log']['value']).isin([5, 10])
    features = [isin]
    df = ft.calculate_feature_matrix(entityset=es, features=features, instance_ids=range(8))
    true = [False, True, True, False, False, False, False, False]
    v = df[isin.get_name()].values.tolist()
    assert true == v


def test_isnull_feat(es):
    value = ft.Feature(es['log']['value'])
    diff = ft.Feature([value, es['log']['session_id']], primitive=Diff)
    isnull = ft.Feature(diff, primitive=IsNull)
    features = [isnull]
    df = ft.calculate_feature_matrix(entityset=es, features=features, instance_ids=range(15))
    # correct_vals_diff = [
    #     np.nan, 5, 5, 5, 5, np.nan, 1, 1, 1, np.nan, np.nan, 5, np.nan, 7, 7]
    correct_vals = [True, False, False, False, False, True, False, False,
                    False, True, True, False, True, False, False]
    values = df[isnull.get_name()].values.tolist()
    assert correct_vals == values


def test_percentile(es):
    v = ft.Feature(es['log']['value'])
    p = ft.Feature(v, primitive=Percentile)
    pandas_backend = PandasBackend(es, [p])
    df = pandas_backend.calculate_all_features(range(10, 17), None)
    true = es['log'].df[v.get_name()].rank(pct=True)
    true = true.loc[range(10, 17)]
    for t, a in zip(true.values, df[p.get_name()].values):
        assert (pd.isnull(t) and pd.isnull(a)) or t == a


def test_dependent_percentile(es):
    v = ft.Feature(es['log']['value'])
    p = ft.Feature(v, primitive=Percentile)
    p2 = ft.Feature(p - 1, primitive=Percentile)
    pandas_backend = PandasBackend(es, [p, p2])
    df = pandas_backend.calculate_all_features(range(10, 17), None)
    true = es['log'].df[v.get_name()].rank(pct=True)
    true = true.loc[range(10, 17)]
    for t, a in zip(true.values, df[p.get_name()].values):
        assert (pd.isnull(t) and pd.isnull(a)) or t == a


def test_agg_percentile(es):
    v = ft.Feature(es['log']['value'])
    p = ft.Feature(v, primitive=Percentile)
    agg = ft.Feature(p, parent_entity=es['sessions'], primitive=Sum)
    pandas_backend = PandasBackend(es, [agg])
    df = pandas_backend.calculate_all_features([0, 1], None)

    log_vals = es['log'].df[[v.get_name(), 'session_id']]
    log_vals['percentile'] = log_vals[v.get_name()].rank(pct=True)
    true_p = log_vals.groupby('session_id')['percentile'].sum()[[0, 1]]
    for t, a in zip(true_p.values, df[agg.get_name()].values):
        assert (pd.isnull(t) and pd.isnull(a)) or t == a


def test_percentile_agg_percentile(es):
    v = ft.Feature(es['log']['value'])
    p = ft.Feature(v, primitive=Percentile)
    agg = ft.Feature(p, parent_entity=es['sessions'], primitive=Sum)
    pagg = ft.Feature(agg, primitive=Percentile)
    pandas_backend = PandasBackend(es, [pagg])
    df = pandas_backend.calculate_all_features([0, 1], None)

    log_vals = es['log'].df[[v.get_name(), 'session_id']]
    log_vals['percentile'] = log_vals[v.get_name()].rank(pct=True)
    true_p = log_vals.groupby('session_id')['percentile'].sum().fillna(0)
    true_p = true_p.rank(pct=True)[[0, 1]]

    for t, a in zip(true_p.values, df[pagg.get_name()].values):
        assert (pd.isnull(t) and pd.isnull(a)) or t == a


def test_percentile_agg(es):
    v = ft.Feature(es['log']['value'])
    agg = ft.Feature(v, parent_entity=es['sessions'], primitive=Sum)
    pagg = ft.Feature(agg, primitive=Percentile)
    pandas_backend = PandasBackend(es, [pagg])
    df = pandas_backend.calculate_all_features([0, 1], None)

    log_vals = es['log'].df[[v.get_name(), 'session_id']]
    true_p = log_vals.groupby('session_id')[v.get_name()].sum().fillna(0)
    true_p = true_p.rank(pct=True)[[0, 1]]

    for t, a in zip(true_p.values, df[pagg.get_name()].values):
        assert (pd.isnull(t) and pd.isnull(a)) or t == a


def test_direct_percentile(es):
    v = ft.Feature(es['customers']['age'])
    p = ft.Feature(v, primitive=Percentile)
    d = ft.Feature(p, es['sessions'])
    pandas_backend = PandasBackend(es, [d])
    df = pandas_backend.calculate_all_features([0, 1], None)

    cust_vals = es['customers'].df[[v.get_name()]]
    cust_vals['percentile'] = cust_vals[v.get_name()].rank(pct=True)
    true_p = cust_vals['percentile'].loc[[0, 0]]
    for t, a in zip(true_p.values, df[d.get_name()].values):
        assert (pd.isnull(t) and pd.isnull(a)) or t == a


def test_direct_agg_percentile(es):
    v = ft.Feature(es['log']['value'])
    p = ft.Feature(v, primitive=Percentile)
    agg = ft.Feature(p, parent_entity=es['customers'], primitive=Sum)
    d = ft.Feature(agg, es['sessions'])
    pandas_backend = PandasBackend(es, [d])
    df = pandas_backend.calculate_all_features([0, 1], None)

    log_vals = es['log'].df[[v.get_name(), 'session_id']]
    log_vals['percentile'] = log_vals[v.get_name()].rank(pct=True)
    log_vals['customer_id'] = [0] * 10 + [1] * 5 + [2] * 2
    true_p = log_vals.groupby('customer_id')['percentile'].sum().fillna(0)
    true_p = true_p[[0, 0]]
    for t, a in zip(true_p.values, df[d.get_name()].values):
        assert (pd.isnull(t) and pd.isnull(a)) or round(t, 3) == round(a, 3)


def test_percentile_with_cutoff(es):
    v = ft.Feature(es['log']['value'])
    p = ft.Feature(v, primitive=Percentile)
    pandas_backend = PandasBackend(es, [p])
    df = pandas_backend.calculate_all_features(
        [2], pd.Timestamp('2011/04/09 10:30:13'))
    assert df[p.get_name()].tolist()[0] == 1.0


def test_two_kinds_of_dependents(es):
    v = ft.Feature(es['log']['value'])
    product = ft.Feature(es['log']['product_id'])
    agg = ft.Feature(v, parent_entity=es['customers'], where=product == 'coke zero', primitive=Sum)
    p = ft.Feature(agg, primitive=Percentile)
    g = ft.Feature(agg, primitive=Absolute)
    agg2 = ft.Feature(v, parent_entity=es['sessions'], where=product == 'coke zero', primitive=Sum)
    # Adding this feature in tests line 218 in pandas_backend
    # where we remove columns in result_frame that already exist
    # in the output entity_frames in preparation for pd.concat
    # In a prior version, this failed because we changed the result_frame
    # variable itself, rather than making a new variable _result_frame.
    # When len(output_frames) > 1, the second iteration won't have
    # all the necessary columns because they were removed in the first
    agg3 = ft.Feature(agg2, parent_entity=es['customers'], primitive=Sum)
    pandas_backend = PandasBackend(es, [p, g, agg3])
    df = pandas_backend.calculate_all_features([0, 1], None)
    assert df[p.get_name()].tolist() == [2. / 3, 1.0]
    assert df[g.get_name()].tolist() == [15, 26]


# P TODO: reimplement like
# def test_like_feat(es):
#     like = Like(es['log']['product_id'], "coke")
#     features = [like]
#     pandas_backend = PandasBackend(es, features)
#     df = pandas_backend.calculate_all_features(range(5), None)
#     true = [True, True, True, False, False]
#     v = df[like.get_name()].values.tolist()
#     assert true == v


# P TODO: reimplement like
# def test_like_feat_other_syntax(es):
#     like = ft.Feature(es['log']['product_id']).LIKE("coke")
#     features = [like]
#     pandas_backend = PandasBackend(es, features)
#     df = pandas_backend.calculate_all_features(range(5), None)
#     true = [True, True, True, False, False]
#     v = df[like.get_name()].values.tolist()
#     assert true == v

def test_make_transform_restricts_time_keyword():
    make_trans_primitive(
        lambda x, time=False: x,
        [Datetime],
        Numeric,
        name="AllowedPrimitive",
        description="This primitive should be accepted",
        uses_calc_time=True)

    error_text = "'time' is a restricted keyword.  Please use a different keyword."
    with pytest.raises(ValueError, match=error_text):
        make_trans_primitive(
            lambda x, time=False: x,
            [Datetime],
            Numeric,
            name="BadPrimitive",
            description="This primitive should error")


def test_make_transform_restricts_time_arg():
    make_trans_primitive(
        lambda time: time,
        [Datetime],
        Numeric,
        name="AllowedPrimitive",
        description="This primitive should be accepted",
        uses_calc_time=True)

    error_text = "'time' is a restricted keyword.  Please use a different keyword."
    with pytest.raises(ValueError, match=error_text):
        make_trans_primitive(
            lambda time: time,
            [Datetime],
            Numeric,
            name="BadPrimitive",
            description="This primitive should erorr")


def test_make_transform_sets_kwargs_correctly(es):
    def pd_is_in(array, list_of_outputs=None):
        if list_of_outputs is None:
            list_of_outputs = []
        return pd.Series(array).isin(list_of_outputs)

    def isin_generate_name(self, base_feature_names):
        return u"%s.isin(%s)" % (base_feature_names[0],
                                 str(self.kwargs['list_of_outputs']))

    IsIn = make_trans_primitive(
        pd_is_in,
        [Variable],
        Boolean,
        name="is_in",
        description="For each value of the base feature, checks whether it is "
        "in a list that is provided.",
        cls_attributes={"generate_name": isin_generate_name})

    isin_1_list = ["toothpaste", "coke_zero"]
    isin_1_base_f = ft.Feature(es['log']['product_id'])
    isin_1 = ft.Feature(isin_1_base_f, primitive=IsIn(list_of_outputs=isin_1_list))
    isin_2_list = ["coke_zero"]
    isin_2_base_f = ft.Feature(es['log']['session_id'])
    isin_2 = ft.Feature(isin_2_base_f, primitive=IsIn(list_of_outputs=isin_2_list))
    assert isin_1_base_f == isin_1.base_features[0]
    assert isin_1_list == isin_1.primitive.kwargs['list_of_outputs']
    assert isin_2_base_f == isin_2.base_features[0]
    assert isin_2_list == isin_2.primitive.kwargs['list_of_outputs']


def test_make_transform_multiple_output_features(es):
    def test_f(x):
        times = pd.Series(x)
        units = ["year", "month", "day", "hour", "minute", "second"]
        return [times.apply(lambda x: getattr(x, unit)) for unit in units]

    def gen_feat_names(self):
        subnames = ["Year", "Month", "Day", "Hour", "Minute", "Second"]
        return ["Now.%s(%s)" % (subname, self.base_features[0].get_name())
                for subname in subnames]

    TestTime = make_trans_primitive(
        function=test_f,
        input_types=[Datetime],
        return_type=Numeric,
        number_output_features=6,
        cls_attributes={"get_feature_names": gen_feat_names},
    )

    join_time_split = ft.Feature(es["log"]["datetime"], primitive=TestTime)
    alt_features = [ft.Feature(es["log"]["datetime"], primitive=Year),
                    ft.Feature(es["log"]["datetime"], primitive=Month),
                    ft.Feature(es["log"]["datetime"], primitive=Day),
                    ft.Feature(es["log"]["datetime"], primitive=Hour),
                    ft.Feature(es["log"]["datetime"], primitive=Minute),
                    ft.Feature(es["log"]["datetime"], primitive=Second)]
    fm, fl = ft.dfs(
        entityset=es,
        target_entity="log",
        trans_primitives=[TestTime, Year, Month, Day, Hour, Minute, Second])

    subnames = join_time_split.get_feature_names()
    altnames = [f.get_name() for f in alt_features]
    for col1, col2 in zip(subnames, altnames):
        assert (fm[col1] == fm[col2]).all()

    # check no feature stacked on new primitive
    for feature in fl:
        for base_feature in feature.base_features:
            assert base_feature.hash() != join_time_split.hash()


def test_tranform_stack_agg(es):
    topn = ft.Feature(es['log']['product_id'],
                      parent_entity=es['customers'],
                      primitive=NMostCommon(n=3))
    with pytest.raises(AssertionError):
        ft.Feature(topn, primitive=Percentile)


def test_feature_names_inherit_from_make_trans_primitive():
    # R TODO
    pass


def test_get_filepath(es):
    class Mod4(TransformPrimitive):
        '''Return base feature modulo 4'''
        name = "mod4"
        input_types = [Numeric]
        return_type = Numeric

        def get_function(self):
            filepath = self.get_filepath("featuretools_unit_test_example.csv")
            reference = pd.read_csv(filepath, header=None, squeeze=True)

            def map_to_word(x):
                def _map(x):
                    if pd.isnull(x):
                        return x
                    return reference[int(x) % 4]
                return pd.Series(x).apply(_map)
            return map_to_word

    feat = ft.Feature(es['log']['value'], primitive=Mod4)
    df = ft.calculate_feature_matrix(features=[feat],
                                     entityset=es,
                                     instance_ids=range(17))

    assert pd.isnull(df["MOD4(value)"][15])
    assert df["MOD4(value)"][0] == 0
    assert df["MOD4(value)"][14] == 2

    fm, fl = ft.dfs(entityset=es,
                    target_entity="log",
                    agg_primitives=[],
                    trans_primitives=[Mod4])

    assert fm["MOD4(value)"][0] == 0
    assert fm["MOD4(value)"][14] == 2
    assert pd.isnull(fm["MOD4(value)"][15])
