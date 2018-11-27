# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import pytest

from ..testing_utils import make_ecommerce_entityset

from featuretools import Timedelta
from featuretools.computational_backends import PandasBackend
from featuretools.primitives import (
    Absolute,
    Add,
    Count,
    CumCount,
    CumMax,
    CumMean,
    CumMin,
    CumSum,
    Day,
    Diff,
    Divide,
    Equals,
    GreaterThan,
    GreaterThanEqualTo,
    Haversine,
    Hour,
    IdentityFeature,
    IsIn,
    IsNull,
    Latitude,
    LessThan,
    LessThanEqualTo,
    Longitude,
    Mod,
    Mode,
    Multiply,
    Negate,
    Not,
    NotEquals,
    NumCharacters,
    NumWords,
    Percentile,
    Subtract,
    Sum,
    get_transform_primitives
)
from featuretools.primitives.base import (
    DirectFeature,
    Feature,
    make_trans_primitive
)
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


def test_make_trans_feat(es):
    f = Hour(es['log']['datetime'])

    pandas_backend = PandasBackend(es, [f])
    df = pandas_backend.calculate_all_features(instance_ids=[0],
                                               time_last=None)
    v = df[f.get_name()][0]
    assert v == 10


def test_diff(es):
    value = IdentityFeature(es['log']['value'])
    customer_id_feat = \
        DirectFeature(es['sessions']['customer_id'],
                      child_entity=es['log'])
    diff1 = Diff(value, es['log']['session_id'])
    diff2 = Diff(value, customer_id_feat)

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
    diff = Diff(es['stores']['num_square_feet'], es['stores'][u'région_id'])
    pandas_backend = PandasBackend(es, [diff])
    df = pandas_backend.calculate_all_features(instance_ids=[5],
                                               time_last=None)
    assert df.shape[0] == 1
    assert df[diff.get_name()].dropna().shape[0] == 0


def test_compare_of_identity(es):
    to_test = [(Equals, [False, False, True, False]),
               (NotEquals, [True, True, False, True]),
               (LessThan, [True, True, False, False]),
               (LessThanEqualTo, [True, True, True, False]),
               (GreaterThan, [False, False, False, True]),
               (GreaterThanEqualTo, [False, False, True, True])]

    features = []
    for test in to_test:
        features.append(test[0](es['log']['value'], 10))

    pandas_backend = PandasBackend(es, features)
    df = pandas_backend.calculate_all_features(instance_ids=[0, 1, 2, 3],
                                               time_last=None)

    for i, test in enumerate(to_test):
        v = df[features[i].get_name()].values.tolist()
        assert v == test[1]


def test_compare_of_direct(es):
    log_rating = DirectFeature(es['products']['rating'],
                               child_entity=es['log'])
    to_test = [(Equals, [False, False, False, False]),
               (NotEquals, [True, True, True, True]),
               (LessThan, [False, False, False, True]),
               (LessThanEqualTo, [False, False, False, True]),
               (GreaterThan, [True, True, True, False]),
               (GreaterThanEqualTo, [True, True, True, False])]

    features = []
    for test in to_test:
        features.append(test[0](log_rating, 4.5))

    pandas_backend = PandasBackend(es, features)
    df = pandas_backend.calculate_all_features(instance_ids=[0, 1, 2, 3],
                                               time_last=None)

    for i, test in enumerate(to_test):
        v = df[features[i].get_name()].values.tolist()
        assert v == test[1]


def test_compare_of_transform(es):
    day = Day(es['log']['datetime'])
    to_test = [(Equals, [False, True]),
               (NotEquals, [True, False]),
               (LessThan, [True, False]),
               (LessThanEqualTo, [True, True]),
               (GreaterThan, [False, False]),
               (GreaterThanEqualTo, [False, True])]

    features = []
    for test in to_test:
        features.append(test[0](day, 10))

    pandas_backend = PandasBackend(es, features)
    df = pandas_backend.calculate_all_features(instance_ids=[0, 14],
                                               time_last=None)

    for i, test in enumerate(to_test):
        v = df[features[i].get_name()].values.tolist()
        assert v == test[1]


def test_compare_of_agg(es):
    count_logs = Count(es['log']['id'],
                       parent_entity=es['sessions'])

    to_test = [(Equals, [False, False, False, True]),
               (NotEquals, [True, True, True, False]),
               (LessThan, [False, False, True, False]),
               (LessThanEqualTo, [False, False, True, True]),
               (GreaterThan, [True, True, False, False]),
               (GreaterThanEqualTo, [True, True, False, True])]

    features = []
    for test in to_test:
        features.append(test[0](count_logs, 2))

    pandas_backend = PandasBackend(es, features)
    df = pandas_backend.calculate_all_features(instance_ids=[0, 1, 2, 3],
                                               time_last=None)

    for i, test in enumerate(to_test):
        v = df[features[i].get_name()].values.tolist()
        assert v == test[1]


def test_compare_all_nans(es):
    nan_feat = Mode(es['log']['product_id'], es['sessions'])
    compare = nan_feat == 'brown bag'
    # before all data
    time_last = pd.Timestamp('1/1/1993')
    pandas_backend = PandasBackend(es, [nan_feat, compare])
    df = pandas_backend.calculate_all_features(instance_ids=[0, 1, 2],
                                               time_last=time_last)
    assert df[nan_feat.get_name()].dropna().shape[0] == 0
    assert not df[compare.get_name()].any()


def test_arithmetic_of_val(es):
    to_test = [(Add, [2.0, 7.0, 12.0, 17.0], [2.0, 7.0, 12.0, 17.0]),
               (Subtract, [-2.0, 3.0, 8.0, 13.0], [2.0, -3.0, -8.0, -13.0]),
               (Multiply, [0, 10, 20, 30], [0, 10, 20, 30]),
               (Divide, [0, 2.5, 5, 7.5], [np.inf, 0.4, 0.2, 2 / 15.0],
                [np.nan, np.inf, np.inf, np.inf])]

    features = []
    logs = es['log']

    for test in to_test:
        features.append(test[0](logs['value'], 2))
        features.append(test[0](2, logs['value']))

    features.append(Divide(logs['value'], 0))

    pandas_backend = PandasBackend(es, features)
    df = pandas_backend.calculate_all_features(instance_ids=[0, 1, 2, 3],
                                               time_last=None)

    for i, test in enumerate(to_test):
        v = df[features[2 * i].get_name()].values.tolist()
        assert v == test[1]
        v = df[features[2 * i + 1].get_name()].values.tolist()
        assert v == test[2]

    test = to_test[-1][-1]
    v = df[features[-1].get_name()].values.tolist()
    assert (np.isnan(v[0]))
    assert v[1:] == test[1:]


def test_arithmetic_two_vals_fails(es):
    error_text = "one of.*must be an instance,of PrimitiveBase or Variable"
    with pytest.raises(ValueError, match=error_text):
        Add(2, 2)


def test_arithmetic_of_identity(es):
    logs = es['log']

    to_test = [(Add, [0., 7., 14., 21.]),
               (Subtract, [0, 3, 6, 9]),
               (Multiply, [0, 10, 40, 90]),
               (Divide, [np.nan, 2.5, 2.5, 2.5])]

    features = []
    for test in to_test:
        features.append(test[0](logs['value'], logs['value_2']))

    pandas_backend = PandasBackend(es, features)
    df = pandas_backend.calculate_all_features(instance_ids=[0, 1, 2, 3],
                                               time_last=None)

    for i, test in enumerate(to_test[:-1]):
        v = df[features[i].get_name()].values.tolist()
        assert v == test[1]
    i, test = 3, to_test[-1]
    v = df[features[i].get_name()].values.tolist()
    assert (np.isnan(v[0]))
    assert v[1:] == test[1][1:]


def test_arithmetic_of_direct(es):
    rating = es['products']['rating']
    log_rating = DirectFeature(rating,
                               child_entity=es['log'])
    customer_age = es['customers']['age']
    session_age = DirectFeature(customer_age,
                                child_entity=es['sessions'])
    log_age = DirectFeature(session_age,
                            child_entity=es['log'])

    to_test = [(Add, [38, 37, 37.5, 37.5]),
               (Subtract, [28, 29, 28.5, 28.5]),
               (Multiply, [165, 132, 148.5, 148.5]),
               (Divide, [6.6, 8.25, 22. / 3, 22. / 3])]

    features = []
    for test in to_test:
        features.append(test[0](log_age, log_rating))

    pandas_backend = PandasBackend(es, features)
    df = pandas_backend.calculate_all_features(instance_ids=[0, 3, 5, 7],
                                               time_last=None)

    for i, test in enumerate(to_test):
        v = df[features[i].get_name()].values.tolist()
        assert v == test[1]


# P TODO: rewrite this  test
def test_arithmetic_of_transform(es):
    diff1 = Diff(IdentityFeature(es['log']['value']),
                 IdentityFeature(es['log']['product_id']))
    diff2 = Diff(IdentityFeature(es['log']['value_2']),
                 IdentityFeature(es['log']['product_id']))

    to_test = [(Add, [np.nan, 14., -7., 3.]),
               (Subtract, [np.nan, 6., -3., 1.]),
               (Multiply, [np.nan, 40., 10., 2.]),
               (Divide, [np.nan, 2.5, 2.5, 2.])]

    features = []
    for test in to_test:
        features.append(test[0](diff1, diff2))

    pandas_backend = PandasBackend(es, features)
    df = pandas_backend.calculate_all_features(instance_ids=[0, 2, 11, 13],
                                               time_last=None)
    for i, test in enumerate(to_test):
        v = df[features[i].get_name()].values.tolist()
        assert np.isnan(v.pop(0))
        assert np.isnan(test[1].pop(0))
        assert v == test[1]


def test_not_feature(es):
    likes_ice_cream = es['customers']['loves_ice_cream']
    not_feat = Not(likes_ice_cream)
    features = [not_feat]
    pandas_backend = PandasBackend(es, features)
    df = pandas_backend.calculate_all_features(instance_ids=[0, 1],
                                               time_last=None)
    v = df[not_feat.get_name()].values
    assert not v[0]
    assert v[1]


def test_arithmetic_of_agg(es):
    customer_id_feat = es['customers']['id']
    store_id_feat = es['stores']['id']
    count_customer = Count(customer_id_feat,
                           parent_entity=es[u'régions'])
    count_stores = Count(store_id_feat,
                         parent_entity=es[u'régions'])
    to_test = [(Add, [6, 2]),
               (Subtract, [0, -2]),
               (Multiply, [9, 0]),
               (Divide, [1, 0])]

    features = []
    for test in to_test:
        features.append(test[0](count_customer, count_stores))

    pandas_backend = PandasBackend(es, features)
    df = pandas_backend.calculate_all_features(
        instance_ids=['United States', 'Mexico'], time_last=None)

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
    latitude = Latitude(log_latlong_feat)
    longitude = Longitude(log_latlong_feat)
    features = [latitude, longitude]
    pandas_backend = PandasBackend(es, features)
    df = pandas_backend.calculate_all_features(instance_ids=range(15),
                                               time_last=None)
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
    haversine = Haversine(log_latlong_feat, log_latlong_feat2)
    features = [haversine]
    pandas_backend = PandasBackend(es, features)
    df = pandas_backend.calculate_all_features(instance_ids=range(15),
                                               time_last=None)
    values = df[haversine.get_name()].values
    real = [0., 524.15585776, 1043.00845747, 1551.12130243,
            2042.79840241, 0., 137.86000883, 275.59396684,
            413.07563177, 0., 0., 524.15585776,
            0., 739.93819145, 1464.27975511]
    assert len(values) == 15
    for i, v in enumerate(real):
        assert v - values[i] < .0001


def test_cum_sum(es):
    log_value_feat = es['log']['value']
    cum_sum = CumSum(log_value_feat, es['log']['session_id'])
    features = [cum_sum]
    pandas_backend = PandasBackend(es, features)
    df = pandas_backend.calculate_all_features(instance_ids=range(15),
                                               time_last=None)
    cvalues = df[cum_sum.get_name()].values
    assert len(cvalues) == 15
    cum_sum_values = [0, 5, 15, 30, 50, 0, 1, 3, 6, 0, 0, 5, 0, 7, 21]
    for i, v in enumerate(cum_sum_values):
        assert v == cvalues[i]


def test_cum_min(es):
    log_value_feat = es['log']['value']
    cum_min = CumMin(log_value_feat, es['log']['session_id'])
    features = [cum_min]
    pandas_backend = PandasBackend(es, features)
    df = pandas_backend.calculate_all_features(instance_ids=range(15),
                                               time_last=None)
    cvalues = df[cum_min.get_name()].values
    assert len(cvalues) == 15
    cum_min_values = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    for i, v in enumerate(cum_min_values):
        assert v == cvalues[i]


def test_cum_max(es):
    log_value_feat = es['log']['value']
    cum_max = CumMax(log_value_feat, es['log']['session_id'])
    features = [cum_max]
    pandas_backend = PandasBackend(es, features)
    df = pandas_backend.calculate_all_features(instance_ids=range(15),
                                               time_last=None)
    cvalues = df[cum_max.get_name()].values
    assert len(cvalues) == 15
    cum_max_values = [0, 5, 10, 15, 20, 0, 1, 2, 3, 0, 0, 5, 0, 7, 14]
    for i, v in enumerate(cum_max_values):
        assert v == cvalues[i]


def test_cum_sum_use_previous(es):
    log_value_feat = es['log']['value']
    cum_sum = CumSum(log_value_feat, es['log']['session_id'],
                     use_previous=Timedelta(3, 'observations',
                                            entity=es['log']))
    features = [cum_sum]
    pandas_backend = PandasBackend(es, features)
    df = pandas_backend.calculate_all_features(instance_ids=range(15),
                                               time_last=None)
    cvalues = df[cum_sum.get_name()].values
    assert len(cvalues) == 15
    cum_sum_values = [0, 5, 15, 30, 45, 0, 1, 3, 6, 0, 0, 5, 0, 7, 21]
    for i, v in enumerate(cum_sum_values):
        assert v == cvalues[i]


def test_cum_sum_use_previous_integer_time(int_es):
    es = int_es

    log_value_feat = es['log']['value']
    with pytest.raises(AssertionError, match=''):
        CumSum(log_value_feat, es['log']['session_id'],
               use_previous=Timedelta(3, 'm'))

    cum_sum = CumSum(log_value_feat, es['log']['session_id'],
                     use_previous=Timedelta(3, 'observations',
                                            entity=es['log']))
    features = [cum_sum]
    pandas_backend = PandasBackend(es, features)
    df = pandas_backend.calculate_all_features(instance_ids=range(15),
                                               time_last=None)
    cvalues = df[cum_sum.get_name()].values
    assert len(cvalues) == 15
    cum_sum_values = [0, 5, 15, 30, 45, 0, 1, 3, 6, 0, 0, 5, 0, 7, 21]
    for i, v in enumerate(cum_sum_values):
        assert v == cvalues[i]


def test_cum_sum_where(es):
    log_value_feat = es['log']['value']
    compare_feat = GreaterThan(log_value_feat, 3)
    dfeat = Feature(es['sessions']['customer_id'], es['log'])
    cum_sum = CumSum(log_value_feat, dfeat,
                     where=compare_feat)
    features = [cum_sum]
    pandas_backend = PandasBackend(es, features)
    df = pandas_backend.calculate_all_features(instance_ids=range(15),
                                               time_last=None)
    cvalues = df[cum_sum.get_name()].values
    assert len(cvalues) == 15
    cum_sum_values = [0, 5, 15, 30, 50, 50, 50, 50, 50, 50,
                      0, 5, 5, 12, 26]
    for i, v in enumerate(cum_sum_values):
        if not np.isnan(v):
            assert v == cvalues[i]
        else:
            assert (np.isnan(cvalues[i]))


def test_cum_sum_use_previous_and_where(es):
    log_value_feat = es['log']['value']
    compare_feat = GreaterThan(log_value_feat, 3)
    # todo should this be cummean?
    dfeat = Feature(es['sessions']['customer_id'], es['log'])
    cum_sum = CumSum(log_value_feat, dfeat,
                     where=compare_feat,
                     use_previous=Timedelta(3, 'observations',
                                            entity=es['log']))
    features = [cum_sum]
    pandas_backend = PandasBackend(es, features)
    df = pandas_backend.calculate_all_features(instance_ids=range(15),
                                               time_last=None)

    cum_sum_values = [0, 5, 15, 30, 45, 45, 45, 45, 45, 45,
                      0, 5, 5, 12, 26]
    cvalues = df[cum_sum.get_name()].values
    assert len(cvalues) == 15
    for i, v in enumerate(cum_sum_values):
        assert v == cvalues[i]


def test_cum_sum_group_on_nan(es):
    log_value_feat = es['log']['value']
    es['log'].df['product_id'] = (['coke zero'] * 3 + ['car'] * 2 +
                                  ['toothpaste'] * 3 + ['brown bag'] * 2 +
                                  ['shoes'] +
                                  [np.nan] * 4 +
                                  ['coke_zero'] * 2)
    cum_sum = CumSum(log_value_feat, es['log']['product_id'])
    features = [cum_sum]
    pandas_backend = PandasBackend(es, features)
    df = pandas_backend.calculate_all_features(instance_ids=range(15),
                                               time_last=None)
    cvalues = df[cum_sum.get_name()].values
    assert len(cvalues) == 15
    cum_sum_values = [0, 5, 15,
                      15, 35,
                      0, 1, 3,
                      3, 3,
                      0,
                      np.nan, np.nan, np.nan, np.nan]
    for i, v in enumerate(cum_sum_values):
        if np.isnan(v):
            assert (np.isnan(cvalues[i]))
        else:
            assert v == cvalues[i]


def test_cum_sum_use_previous_group_on_nan(es):
    # TODO: Figure out how to test where `df`
    # in pd_rolling get_function() has multiindex
    log_value_feat = es['log']['value']
    es['log'].df['product_id'] = (['coke zero'] * 3 + ['car'] * 2 +
                                  ['toothpaste'] * 3 + ['brown bag'] * 2 +
                                  ['shoes'] +
                                  [np.nan] * 4 +
                                  ['coke_zero'] * 2)
    cum_sum = CumSum(log_value_feat,
                     es['log']['product_id'],
                     es["log"]["datetime"],
                     use_previous=Timedelta(40, 'seconds'))
    features = [cum_sum]
    pandas_backend = PandasBackend(es, features)
    df = pandas_backend.calculate_all_features(instance_ids=range(15),
                                               time_last=None)
    cvalues = df[cum_sum.get_name()].values
    assert len(cvalues) == 15
    cum_sum_values = [0, 5, 15,
                      15, 35,
                      0, 1, 3,
                      3, 0,
                      0,
                      np.nan, np.nan, np.nan, np.nan]
    for i, v in enumerate(cum_sum_values):
        if np.isnan(v):
            assert (np.isnan(cvalues[i]))
        else:
            assert v == cvalues[i]


def test_cum_sum_use_previous_and_where_absolute(es):
    log_value_feat = es['log']['value']
    compare_feat = GreaterThan(log_value_feat, 3)
    dfeat = Feature(es['sessions']['customer_id'], es['log'])
    cum_sum = CumSum(log_value_feat, dfeat, es["log"]["datetime"],
                     where=compare_feat,
                     use_previous=Timedelta(40, 'seconds'))
    features = [cum_sum]
    pandas_backend = PandasBackend(es, features)
    df = pandas_backend.calculate_all_features(instance_ids=range(15),
                                               time_last=None)

    cum_sum_values = [0, 5, 15, 30, 50, 0, 0, 0, 0, 0,
                      0, 5, 0, 7, 21]
    cvalues = df[cum_sum.get_name()].values
    assert len(cvalues) == 15
    for i, v in enumerate(cum_sum_values):
        assert v == cvalues[i]


def test_cum_handles_uses_full_entity(es):
    def check(feature):
        pandas_backend = PandasBackend(es, [feature])
        df_1 = pandas_backend.calculate_all_features(instance_ids=[0, 1, 2], time_last=None)
        df_2 = pandas_backend.calculate_all_features(instance_ids=[2], time_last=None)

        # check that the value for instance id 2 matches
        assert (df_2.loc[2] == df_1.loc[2]).all()

    for primitive in [CumSum, CumMean, CumMax, CumMin]:
        check(primitive(es['log']['value'], es['log']['session_id']))

    check(CumCount(es['log']['id'], es['log']['session_id']))


def test_cum_mean(es):
    log_value_feat = es['log']['value']
    cum_mean = CumMean(log_value_feat, es['log']['session_id'])
    features = [cum_mean]
    pandas_backend = PandasBackend(es, features)
    df = pandas_backend.calculate_all_features(instance_ids=range(15),
                                               time_last=None)
    cvalues = df[cum_mean.get_name()].values
    assert len(cvalues) == 15
    cum_mean_values = [0, 2.5, 5, 7.5, 10, 0, .5, 1, 1.5, 0, 0, 2.5, 0, 3.5, 7]
    for i, v in enumerate(cum_mean_values):
        assert v == cvalues[i]


def test_cum_mean_use_previous(es):
    log_value_feat = es['log']['value']
    cum_mean = CumMean(log_value_feat, es['log']['session_id'],
                       use_previous=Timedelta(3, 'observations',
                                              entity=es['log']))
    features = [cum_mean]
    pandas_backend = PandasBackend(es, features)
    df = pandas_backend.calculate_all_features(instance_ids=range(15),
                                               time_last=None)
    cvalues = df[cum_mean.get_name()].values
    assert len(cvalues) == 15
    cum_mean_values = [0, 2.5, 5, 10, 15, 0, .5, 1, 2, 0, 0, 2.5, 0, 3.5, 7]
    for i, v in enumerate(cum_mean_values):
        assert v == cvalues[i]


def test_cum_mean_where(es):
    log_value_feat = es['log']['value']
    compare_feat = GreaterThan(log_value_feat, 3)
    dfeat = Feature(es['sessions']['customer_id'], es['log'])
    cum_mean = CumMean(log_value_feat, dfeat,
                       where=compare_feat)
    features = [cum_mean]
    pandas_backend = PandasBackend(es, features)
    df = pandas_backend.calculate_all_features(instance_ids=range(15),
                                               time_last=None)
    cvalues = df[cum_mean.get_name()].values
    assert len(cvalues) == 15
    cum_mean_values = [0, 5, 7.5, 10, 12.5, 12.5, 12.5, 12.5, 12.5, 12.5,
                       0, 5, 5, 6, 26. / 3]

    for i, v in enumerate(cum_mean_values):
        if not np.isnan(v):
            assert v == cvalues[i]
        else:
            assert (np.isnan(cvalues[i]))


def test_cum_mean_use_previous_and_where(es):
    log_value_feat = es['log']['value']
    compare_feat = GreaterThan(log_value_feat, 3)
    # todo should this be cummean?
    dfeat = Feature(es['sessions']['customer_id'], es['log'])
    cum_mean = CumMean(log_value_feat, dfeat,
                       where=compare_feat,
                       use_previous=Timedelta(2, 'observations',
                                              entity=es['log']))
    features = [cum_mean]
    pandas_backend = PandasBackend(es, features)
    df = pandas_backend.calculate_all_features(instance_ids=range(15),
                                               time_last=None)

    cum_mean_values = [0, 5, 7.5, 12.5, 17.5, 17.5, 17.5, 17.5, 17.5, 17.5,
                       0, 5, 5, 6, 10.5]
    cvalues = df[cum_mean.get_name()].values
    assert len(cvalues) == 15
    for i, v in enumerate(cum_mean_values):
        assert v == cvalues[i]


def test_cum_count(es):
    log_id_feat = es['log']['id']
    cum_count = CumCount(log_id_feat, es['log']['session_id'])
    features = [cum_count]
    pandas_backend = PandasBackend(es, features)
    df = pandas_backend.calculate_all_features(instance_ids=range(15),
                                               time_last=None)
    cvalues = df[cum_count.get_name()].values
    assert len(cvalues) == 15
    cum_count_values = [1, 2, 3, 4, 5, 1, 2, 3, 4, 1, 1, 2, 1, 2, 3]
    for i, v in enumerate(cum_count_values):
        assert v == cvalues[i]


def test_text_primitives(es):
    words = NumWords(es['log']['comments'])
    chars = NumCharacters(es['log']['comments'])

    features = [words, chars]
    pandas_backend = PandasBackend(es, features)
    df = pandas_backend.calculate_all_features(instance_ids=range(15),
                                               time_last=None)

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


def test_overrides(es):
    value = Feature(es['log']['value'])
    value2 = Feature(es['log']['value_2'])

    feats = [Add, Subtract, Multiply, Divide]
    compare_ops = [GreaterThan, LessThan, Equals, NotEquals,
                   GreaterThanEqualTo, LessThanEqualTo]
    assert Negate(value).hash() == (-value).hash()

    compares = [(value, value),
                (value, value2),
                (value2, 2)]
    overrides = [
        value + value,
        value - value,
        value * value,
        value / value,
        value > value,
        value < value,
        value == value,
        value != value,
        value >= value,
        value <= value,

        value + value2,
        value - value2,
        value * value2,
        value / value2,
        value > value2,
        value < value2,
        value == value2,
        value != value2,
        value >= value2,
        value <= value2,

        value2 + 2,
        value2 - 2,
        value2 * 2,
        value2 / 2,
        value2 > 2,
        value2 < 2,
        value2 == 2,
        value2 != 2,
        value2 >= 2,
        value2 <= 2,
    ]

    i = 0
    for left, right in compares:
        for feat in feats:
            f = feat(left, right)
            o = overrides[i]
            assert o.hash() == f.hash()
            i += 1

        for compare_op in compare_ops:
            f = compare_op(left, right)
            o = overrides[i]
            assert o.hash() == f.hash()
            i += 1

    our_reverse_overrides = [
        2 + value2,
        2 - value2,
        2 * value2,
        2 / value2]
    i = 0
    for feat in feats:
        if feat != Mod:
            f = feat(2, value2)
            o = our_reverse_overrides[i]
            assert o.hash() == f.hash()
            i += 1

    python_reverse_overrides = [
        2 < value2,
        2 > value2,
        2 == value2,
        2 != value2,
        2 <= value2,
        2 >= value2]
    i = 0
    for compare_op in compare_ops:
        f = compare_op(value2, 2)
        o = python_reverse_overrides[i]
        assert o.hash() == f.hash()
        i += 1


def test_override_boolean(es):
    count = Count(es['log']['id'], es['sessions'])
    count_lo = GreaterThan(count, 1)
    count_hi = LessThan(count, 10)

    to_test = [[True, True, True],
               [True, True, False],
               [False, False, True]]

    features = []
    features.append(count_lo.OR(count_hi))
    features.append(count_lo.AND(count_hi))
    features.append(~(count_lo.AND(count_hi)))

    pandas_backend = PandasBackend(es, features)
    df = pandas_backend.calculate_all_features(instance_ids=[0, 1, 2],
                                               time_last=None)
    for i, test in enumerate(to_test):
        v = df[features[i].get_name()].values.tolist()
        assert v == test


def test_override_cmp_from_variable(es):
    count_lo = IdentityFeature(es['log']['value']) > 1

    to_test = [False, True, True]

    features = [count_lo]

    pandas_backend = PandasBackend(es, features)
    df = pandas_backend.calculate_all_features(instance_ids=[0, 1, 2],
                                               time_last=None)
    v = df[count_lo.get_name()].values.tolist()
    for i, test in enumerate(to_test):
        assert v[i] == test


def test_override_cmp(es):
    count = Count(es['log']['id'], es['sessions'])
    _sum = Sum(es['log']['value'], es['sessions'])
    gt_lo = count > 1
    gt_other = count > _sum
    ge_lo = count >= 1
    ge_other = count >= _sum
    lt_hi = count < 10
    lt_other = count < _sum
    le_hi = count <= 10
    le_other = count <= _sum
    ne_lo = count != 1
    ne_other = count != _sum

    to_test = [[True, True, False],
               [False, False, True],
               [True, True, True],
               [False, False, True],
               [True, True, True],
               [True, True, False],
               [True, True, True],
               [True, True, False]]
    features = [gt_lo, gt_other, ge_lo, ge_other, lt_hi,
                lt_other, le_hi, le_other, ne_lo, ne_other]
    pandas_backend = PandasBackend(es, features)
    df = pandas_backend.calculate_all_features(instance_ids=[0, 1, 2],
                                               time_last=None)
    for i, test in enumerate(to_test):
        v = df[features[i].get_name()].values.tolist()
        assert v == test


def test_isin_feat(es):
    isin = IsIn(es['log']['product_id'],
                list_of_outputs=["toothpaste", "coke zero"])
    features = [isin]
    pandas_backend = PandasBackend(es, features)
    df = pandas_backend.calculate_all_features(range(8), None)
    true = [True, True, True, False, False, True, True, True]
    v = df[isin.get_name()].values.tolist()
    assert true == v


def test_isin_feat_other_syntax(es):
    isin = Feature(es['log']['product_id']).isin(["toothpaste", "coke zero"])
    features = [isin]
    pandas_backend = PandasBackend(es, features)
    df = pandas_backend.calculate_all_features(range(8), None)
    true = [True, True, True, False, False, True, True, True]
    v = df[isin.get_name()].values.tolist()
    assert true == v


def test_isin_feat_other_syntax_int(es):
    isin = Feature(es['log']['value']).isin([5, 10])
    features = [isin]
    pandas_backend = PandasBackend(es, features)
    df = pandas_backend.calculate_all_features(range(8), None)
    true = [False, True, True, False, False, False, False, False]
    v = df[isin.get_name()].values.tolist()
    assert true == v


def test_isin_feat_custom(es):
    def pd_is_in(array, list_of_outputs=None):
        if list_of_outputs is None:
            list_of_outputs = []
        return pd.Series(array).isin(list_of_outputs)

    def isin_generate_name(self):
        return u"%s.isin(%s)" % (self.base_features[0].get_name(),
                                 str(self.kwargs['list_of_outputs']))

    IsIn = make_trans_primitive(
        pd_is_in,
        [Variable],
        Boolean,
        name="is_in",
        description="For each value of the base feature, checks whether it is "
        "in a list that is provided.",
        cls_attributes={"generate_name": isin_generate_name})

    isin = IsIn(es['log']['product_id'],
                list_of_outputs=["toothpaste", "coke zero"])
    features = [isin]
    pandas_backend = PandasBackend(es, features)
    df = pandas_backend.calculate_all_features(range(8), None)
    true = [True, True, True, False, False, True, True, True]
    v = df[isin.get_name()].values.tolist()
    assert true == v

    isin = Feature(es['log']['product_id']).isin(["toothpaste", "coke zero"])
    features = [isin]
    pandas_backend = PandasBackend(es, features)
    df = pandas_backend.calculate_all_features(range(8), None)
    true = [True, True, True, False, False, True, True, True]
    v = df[isin.get_name()].values.tolist()
    assert true == v

    isin = Feature(es['log']['value']).isin([5, 10])
    features = [isin]
    pandas_backend = PandasBackend(es, features)
    df = pandas_backend.calculate_all_features(range(8), None)
    true = [False, True, True, False, False, False, False, False]
    v = df[isin.get_name()].values.tolist()
    assert true == v


def test_isnull_feat(es):
    value = IdentityFeature(es['log']['value'])
    diff = Diff(value, es['log']['session_id'])
    isnull = IsNull(diff)
    features = [isnull]
    pandas_backend = PandasBackend(es, features)
    df = pandas_backend.calculate_all_features(range(15), None)
    # correct_vals_diff = [
    #     np.nan, 5, 5, 5, 5, np.nan, 1, 1, 1, np.nan, np.nan, 5, np.nan, 7, 7]
    correct_vals = [True, False, False, False, False, True, False, False,
                    False, True, True, False, True, False, False]
    values = df[isnull.get_name()].values.tolist()
    assert correct_vals == values


def test_init_and_name(es):
    from featuretools import calculate_feature_matrix
    log = es['log']
    features = [Feature(v) for v in log.variables] +\
        [GreaterThan(Feature(es["products"]["rating"], es["log"]), 2.5)]
    # Add Timedelta feature
    features.append(pd.Timestamp.now() - Feature(log['datetime']))
    for transform_prim in get_transform_primitives().values():
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
            instance = transform_prim(*s)

            # try to get name and calculate
            instance.get_name()
            calculate_feature_matrix([instance], entityset=es).head(5)


def test_percentile(es):
    v = Feature(es['log']['value'])
    p = Percentile(v)
    pandas_backend = PandasBackend(es, [p])
    df = pandas_backend.calculate_all_features(range(10, 17), None)
    true = es['log'].df[v.get_name()].rank(pct=True)
    true = true.loc[range(10, 17)]
    for t, a in zip(true.values, df[p.get_name()].values):
        assert (pd.isnull(t) and pd.isnull(a)) or t == a


def test_dependent_percentile(es):
    v = Feature(es['log']['value'])
    p = Percentile(v)
    p2 = Percentile(p - 1)
    pandas_backend = PandasBackend(es, [p, p2])
    df = pandas_backend.calculate_all_features(range(10, 17), None)
    true = es['log'].df[v.get_name()].rank(pct=True)
    true = true.loc[range(10, 17)]
    for t, a in zip(true.values, df[p.get_name()].values):
        assert (pd.isnull(t) and pd.isnull(a)) or t == a


def test_agg_percentile(es):
    v = Feature(es['log']['value'])
    p = Percentile(v)
    agg = Sum(p, es['sessions'])
    pandas_backend = PandasBackend(es, [agg])
    df = pandas_backend.calculate_all_features([0, 1], None)

    log_vals = es['log'].df[[v.get_name(), 'session_id']]
    log_vals['percentile'] = log_vals[v.get_name()].rank(pct=True)
    true_p = log_vals.groupby('session_id')['percentile'].sum()[[0, 1]]
    for t, a in zip(true_p.values, df[agg.get_name()].values):
        assert (pd.isnull(t) and pd.isnull(a)) or t == a


def test_percentile_agg_percentile(es):
    v = Feature(es['log']['value'])
    p = Percentile(v)
    agg = Sum(p, es['sessions'])
    pagg = Percentile(agg)
    pandas_backend = PandasBackend(es, [pagg])
    df = pandas_backend.calculate_all_features([0, 1], None)

    log_vals = es['log'].df[[v.get_name(), 'session_id']]
    log_vals['percentile'] = log_vals[v.get_name()].rank(pct=True)
    true_p = log_vals.groupby('session_id')['percentile'].sum().fillna(0)
    true_p = true_p.rank(pct=True)[[0, 1]]

    for t, a in zip(true_p.values, df[pagg.get_name()].values):
        assert (pd.isnull(t) and pd.isnull(a)) or t == a


def test_percentile_agg(es):
    v = Feature(es['log']['value'])
    agg = Sum(v, es['sessions'])
    pagg = Percentile(agg)
    pandas_backend = PandasBackend(es, [pagg])
    df = pandas_backend.calculate_all_features([0, 1], None)

    log_vals = es['log'].df[[v.get_name(), 'session_id']]
    true_p = log_vals.groupby('session_id')[v.get_name()].sum().fillna(0)
    true_p = true_p.rank(pct=True)[[0, 1]]

    for t, a in zip(true_p.values, df[pagg.get_name()].values):
        assert (pd.isnull(t) and pd.isnull(a)) or t == a


def test_direct_percentile(es):
    v = Feature(es['customers']['age'])
    p = Percentile(v)
    d = Feature(p, es['sessions'])
    pandas_backend = PandasBackend(es, [d])
    df = pandas_backend.calculate_all_features([0, 1], None)

    cust_vals = es['customers'].df[[v.get_name()]]
    cust_vals['percentile'] = cust_vals[v.get_name()].rank(pct=True)
    true_p = cust_vals['percentile'].loc[[0, 0]]
    for t, a in zip(true_p.values, df[d.get_name()].values):
        assert (pd.isnull(t) and pd.isnull(a)) or t == a


def test_direct_agg_percentile(es):
    v = Feature(es['log']['value'])
    p = Percentile(v)
    agg = Sum(p, es['customers'])
    d = Feature(agg, es['sessions'])
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
    v = Feature(es['log']['value'])
    p = Percentile(v)
    pandas_backend = PandasBackend(es, [p])
    df = pandas_backend.calculate_all_features(
        [2], pd.Timestamp('2011/04/09 10:30:13'))
    assert df[p.get_name()].tolist()[0] == 1.0


def test_two_kinds_of_dependents(es):
    v = Feature(es['log']['value'])
    product = Feature(es['log']['product_id'])
    agg = Sum(v, es['customers'], where=product == 'coke zero')
    p = Percentile(agg)
    g = Absolute(agg)
    agg2 = Sum(v, es['sessions'], where=product == 'coke zero')
    # Adding this feature in tests line 218 in pandas_backend
    # where we remove columns in result_frame that already exist
    # in the output entity_frames in preparation for pd.concat
    # In a prior version, this failed because we changed the result_frame
    # variable itself, rather than making a new variable _result_frame.
    # When len(output_frames) > 1, the second iteration won't have
    # all the necessary columns because they were removed in the first
    agg3 = Sum(agg2, es['customers'])
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
#     like = Feature(es['log']['product_id']).LIKE("coke")
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

    def isin_generate_name(self):
        return u"%s.isin(%s)" % (self.base_features[0].get_name(),
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
    isin_1_base_f = Feature(es['log']['product_id'])
    isin_1 = IsIn(isin_1_base_f, list_of_outputs=isin_1_list)
    isin_2_list = ["coke_zero"]
    isin_2_base_f = Feature(es['log']['session_id'])
    isin_2 = IsIn(isin_2_base_f, list_of_outputs=isin_2_list)
    assert isin_1_base_f == isin_1.base_features[0]
    assert isin_1_list == isin_1.kwargs['list_of_outputs']
    assert isin_2_base_f == isin_2.base_features[0]
    assert isin_2_list == isin_2.kwargs['list_of_outputs']
