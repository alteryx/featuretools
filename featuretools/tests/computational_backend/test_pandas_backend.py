# -*- coding: utf-8 -*-

from datetime import datetime

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_array_equal

from ..testing_utils import make_ecommerce_entityset

import featuretools as ft
from featuretools import Timedelta
from featuretools.computational_backends.pandas_backend import PandasBackend
from featuretools.feature_base import DirectFeature, IdentityFeature
from featuretools.primitives import (  # NMostCommon,
    And,
    Count,
    EqualScalar,
    GreaterThanEqualToScalar,
    GreaterThanScalar,
    LessThanEqualToScalar,
    LessThanScalar,
    Mean,
    Min,
    Mode,
    NMostCommon,
    NotEqualScalar,
    NumTrue,
    Sum,
    Trend
)
from featuretools.primitives.base import AggregationPrimitive
from featuretools.variable_types import Numeric


@pytest.fixture(scope='module')
def entityset():
    return make_ecommerce_entityset()


@pytest.fixture
def backend(entityset):
    def inner(features):
        return PandasBackend(entityset, features)
    return inner


def test_make_identity(entityset, backend):
    f = IdentityFeature(entityset['log']['datetime'])

    pandas_backend = backend([f])
    df = pandas_backend.calculate_all_features(instance_ids=[0],
                                               time_last=None)
    v = df[f.get_name()][0]
    assert (v == datetime(2011, 4, 9, 10, 30, 0))


def test_make_dfeat(entityset, backend):
    f = DirectFeature(entityset['customers']['age'],
                      child_entity=entityset['sessions'])

    pandas_backend = backend([f])
    df = pandas_backend.calculate_all_features(instance_ids=[0],
                                               time_last=None)
    v = df[f.get_name()][0]
    assert (v == 33)


def test_make_agg_feat_of_identity_variable(entityset, backend):
    agg_feat = ft.Feature(entityset['log']['value'], parent_entity=entityset['sessions'], primitive=Sum)

    pandas_backend = backend([agg_feat])
    df = pandas_backend.calculate_all_features(instance_ids=[0],
                                               time_last=None)
    v = df[agg_feat.get_name()][0]
    assert (v == 50)


def test_make_agg_feat_of_identity_index_variable(entityset, backend):
    agg_feat = ft.Feature(entityset['log']['id'], parent_entity=entityset['sessions'], primitive=Count)

    pandas_backend = backend([agg_feat])
    df = pandas_backend.calculate_all_features(instance_ids=[0],
                                               time_last=None)
    v = df[agg_feat.get_name()][0]
    assert (v == 5)


def test_make_agg_feat_where_count(entityset, backend):
    agg_feat = ft.Feature(entityset['log']['id'],
                          parent_entity=entityset['sessions'],
                          where=IdentityFeature(entityset['log']['product_id']) == 'coke zero',
                          primitive=Count)

    pandas_backend = backend([agg_feat])
    df = pandas_backend.calculate_all_features(instance_ids=[0],
                                               time_last=None)

    v = df[agg_feat.get_name()][0]
    assert (v == 3)


def test_make_agg_feat_using_prev_time(entityset, backend):
    agg_feat = ft.Feature(entityset['log']['id'],
                          parent_entity=entityset['sessions'],
                          use_previous=Timedelta(10, 's'),
                          primitive=Count)

    pandas_backend = backend([agg_feat])
    df = pandas_backend.calculate_all_features(instance_ids=[0],
                                               time_last=datetime(2011, 4, 9, 10, 30, 10))

    v = df[agg_feat.get_name()][0]
    assert (v == 2)

    df = pandas_backend.calculate_all_features(instance_ids=[0],
                                               time_last=datetime(2011, 4, 9, 10, 30, 30))

    v = df[agg_feat.get_name()][0]
    assert (v == 1)


def test_make_agg_feat_using_prev_n_events(entityset, backend):
    agg_feat_1 = ft.Feature(entityset['log']['value'],
                            parent_entity=entityset['sessions'],
                            use_previous=Timedelta(1, 'observations',
                                                   entity=entityset['log']),
                            primitive=Min)

    agg_feat_2 = ft.Feature(entityset['log']['value'],
                            parent_entity=entityset['sessions'],
                            use_previous=Timedelta(3, 'observations',
                                                   entity=entityset['log']),
                            primitive=Min)

    assert agg_feat_1.get_name() != agg_feat_2.get_name(), \
        'Features should have different names based on use_previous'

    pandas_backend = backend([agg_feat_1, agg_feat_2])
    df = pandas_backend.calculate_all_features(instance_ids=[0],
                                               time_last=datetime(2011, 4, 9, 10, 30, 6))

    # time_last is included by default
    v1 = df[agg_feat_1.get_name()][0]
    v2 = df[agg_feat_2.get_name()][0]
    assert v1 == 5
    assert v2 == 0

    df = pandas_backend.calculate_all_features(instance_ids=[0],
                                               time_last=datetime(2011, 4, 9, 10, 30, 30))

    v1 = df[agg_feat_1.get_name()][0]
    v2 = df[agg_feat_2.get_name()][0]
    assert v1 == 20
    assert v2 == 10


def test_make_agg_feat_multiple_dtypes(entityset, backend):
    compare_prod = IdentityFeature(entityset['log']['product_id']) == 'coke zero'

    agg_feat = ft.Feature(entityset['log']['id'],
                          parent_entity=entityset['sessions'],
                          where=compare_prod,
                          primitive=Count)

    agg_feat2 = ft.Feature(entityset['log']['product_id'],
                           parent_entity=entityset['sessions'],
                           where=compare_prod,
                           primitive=Mode)

    pandas_backend = backend([agg_feat, agg_feat2])
    df = pandas_backend.calculate_all_features(instance_ids=[0],
                                               time_last=None)

    v = df[agg_feat.get_name()][0]
    v2 = df[agg_feat2.get_name()][0]
    assert (v == 3)
    assert (v2 == 'coke zero')


def test_make_agg_feat_where_different_identity_feat(entityset, backend):
    feats = []
    where_cmps = [LessThanScalar, GreaterThanScalar, LessThanEqualToScalar,
                  GreaterThanEqualToScalar, EqualScalar, NotEqualScalar]
    for where_cmp in where_cmps:
        feats.append(ft.Feature(entityset['log']['id'],
                                parent_entity=entityset['sessions'],
                                where=ft.Feature(entityset['log']['datetime'], primitive=where_cmp(datetime(2011, 4, 10, 10, 40, 1))),
                                primitive=Count))

    df = ft.calculate_feature_matrix(entityset=entityset, features=feats, instance_ids=[0, 1, 2, 3])

    for i, where_cmp in enumerate(where_cmps):
        name = feats[i].get_name()
        instances = df[name]
        v0, v1, v2, v3 = instances[0:4]
        if where_cmp == LessThanScalar:
            assert (v0 == 5)
            assert (v1 == 4)
            assert (v2 == 1)
            assert (v3 == 1)
        elif where_cmp == GreaterThanScalar:
            assert (v0 == 0)
            assert (v1 == 0)
            assert (v2 == 0)
            assert (v3 == 0)
        elif where_cmp == LessThanEqualToScalar:
            assert (v0 == 5)
            assert (v1 == 4)
            assert (v2 == 1)
            assert (v3 == 2)
        elif where_cmp == GreaterThanEqualToScalar:
            assert (v0 == 0)
            assert (v1 == 0)
            assert (v2 == 0)
            assert (v3 == 1)
        elif where_cmp == EqualScalar:
            assert (v0 == 0)
            assert (v1 == 0)
            assert (v2 == 0)
            assert (v3 == 1)
        elif where_cmp == NotEqualScalar:
            assert (v0 == 5)
            assert (v1 == 4)
            assert (v2 == 1)
            assert (v3 == 1)


def test_make_agg_feat_of_grandchild_entity(entityset, backend):
    agg_feat = ft.Feature(entityset['log']['id'], parent_entity=entityset['customers'], primitive=Count)

    pandas_backend = backend([agg_feat])
    df = pandas_backend.calculate_all_features(instance_ids=[0],
                                               time_last=None)
    v = df[agg_feat.get_name()][0]
    assert (v == 10)


def test_make_agg_feat_where_count_feat(entityset, backend):
    """
    Feature we're creating is:
    Number of sessions for each customer where the
    number of logs in the session is less than 3
    """
    log_count_feat = ft.Feature(entityset['log']['id'], parent_entity=entityset['sessions'], primitive=Count)

    feat = ft.Feature(entityset['sessions']['id'],
                      parent_entity=entityset['customers'],
                      where=log_count_feat > 1,
                      primitive=Count)

    pandas_backend = backend([feat])
    df = pandas_backend.calculate_all_features(instance_ids=[0, 1],
                                               time_last=None)
    name = feat.get_name()
    instances = df[name]
    v0, v1 = instances[0:2]
    assert (v0 == 2)
    assert (v1 == 2)


def test_make_compare_feat(entityset, backend):
    """
    Feature we're creating is:
    Number of sessions for each customer where the
    number of logs in the session is less than 3
    """
    log_count_feat = ft.Feature(entityset['log']['id'], parent_entity=entityset['sessions'], primitive=Count)

    mean_agg_feat = ft.Feature(log_count_feat, parent_entity=entityset['customers'], primitive=Mean)

    mean_feat = DirectFeature(mean_agg_feat, child_entity=entityset['sessions'])

    feat = log_count_feat > mean_feat

    pandas_backend = backend([feat])
    df = pandas_backend.calculate_all_features(instance_ids=[0, 1, 2],
                                               time_last=None)
    name = feat.get_name()
    instances = df[name]
    v0, v1, v2 = instances[0:3]
    assert v0
    assert v1
    assert not v2


def test_make_agg_feat_where_count_and_device_type_feat(entityset, backend):
    """
    Feature we're creating is:
    Number of sessions for each customer where the
    number of logs in the session is less than 3
    """
    log_count_feat = ft.Feature(entityset['log']['id'], parent_entity=entityset['sessions'], primitive=Count)

    compare_count = log_count_feat == 1
    compare_device_type = IdentityFeature(entityset['sessions']['device_type']) == 1
    and_feat = ft.Feature([compare_count, compare_device_type], primitive=And)
    feat = ft.Feature(entityset['sessions']['id'],
                      parent_entity=entityset['customers'],
                      where=and_feat,
                      primitive=Count)

    pandas_backend = backend([feat])
    df = pandas_backend.calculate_all_features(instance_ids=[0],
                                               time_last=None)
    name = feat.get_name()
    instances = df[name]
    assert (instances[0] == 1)


def test_make_agg_feat_where_count_or_device_type_feat(entityset, backend):
    """
    Feature we're creating is:
    Number of sessions for each customer where the
    number of logs in the session is less than 3
    """
    log_count_feat = ft.Feature(entityset['log']['id'], parent_entity=entityset['sessions'], primitive=Count)

    compare_count = log_count_feat > 1
    compare_device_type = IdentityFeature(entityset['sessions']['device_type']) == 1
    or_feat = compare_count.OR(compare_device_type)
    feat = ft.Feature(entityset['sessions']['id'],
                      parent_entity=entityset['customers'],
                      where=or_feat,
                      primitive=Count)

    pandas_backend = backend([feat])
    df = pandas_backend.calculate_all_features(instance_ids=[0],
                                               time_last=None)
    name = feat.get_name()
    instances = df[name]
    assert (instances[0] == 3)


def test_make_agg_feat_of_agg_feat(entityset, backend):
    log_count_feat = ft.Feature(entityset['log']['id'], parent_entity=entityset['sessions'], primitive=Count)

    customer_sum_feat = ft.Feature(log_count_feat, parent_entity=entityset['customers'], primitive=Sum)

    pandas_backend = backend([customer_sum_feat])
    df = pandas_backend.calculate_all_features(instance_ids=[0],
                                               time_last=None)
    v = df[customer_sum_feat.get_name()][0]
    assert (v == 10)


def test_make_dfeat_of_agg_feat_on_self(entityset, backend):
    """
    The graph looks like this:

        R       R = Regions, a parent of customers
        |
        C       C = Customers, the entity we're trying to predict on
        |
       etc.

    We're trying to calculate a DFeat from C to R on an agg_feat of R on C.
    """
    customer_count_feat = ft.Feature(entityset['customers']['id'], parent_entity=entityset[u'régions'], primitive=Count)

    num_customers_feat = DirectFeature(customer_count_feat, child_entity=entityset['customers'])

    pandas_backend = backend([num_customers_feat])
    df = pandas_backend.calculate_all_features(instance_ids=[0],
                                               time_last=None)
    v = df[num_customers_feat.get_name()][0]
    assert (v == 3)


def test_make_dfeat_of_agg_feat_through_parent(entityset, backend):
    """
    The graph looks like this:

        R       C = Customers, the entity we're trying to predict on
       / \\     R = Regions, a parent of customers
      S   C     S = Stores, a child of regions
          |
         etc.

    We're trying to calculate a DFeat from C to R on an agg_feat of R on S.
    """
    store_id_feat = IdentityFeature(entityset['stores']['id'])

    store_count_feat = ft.Feature(store_id_feat, parent_entity=entityset[u'régions'], primitive=Count)

    num_stores_feat = DirectFeature(store_count_feat, child_entity=entityset['customers'])

    pandas_backend = backend([num_stores_feat])
    df = pandas_backend.calculate_all_features(instance_ids=[0],
                                               time_last=None)
    v = df[num_stores_feat.get_name()][0]
    assert (v == 3)


def test_make_deep_agg_feat_of_dfeat_of_agg_feat(entityset, backend):
    """
    The graph looks like this (higher implies parent):

          C     C = Customers, the entity we're trying to predict on
          |     S = Sessions, a child of Customers
      P   S     L = Log, a child of both Sessions and Log
       \\ /     P = Products, a parent of Log which is not a descendent of customers
        L

    We're trying to calculate a DFeat from L to P on an agg_feat of P on L, and
    then aggregate it with another agg_feat of C on L.
    """
    log_count_feat = ft.Feature(entityset['log']['id'], parent_entity=entityset['products'], primitive=Count)

    product_purchases_feat = DirectFeature(log_count_feat,
                                           child_entity=entityset['log'])

    purchase_popularity = ft.Feature(product_purchases_feat, parent_entity=entityset['customers'], primitive=Mean)

    pandas_backend = backend([purchase_popularity])
    df = pandas_backend.calculate_all_features(instance_ids=[0],
                                               time_last=None)
    v = df[purchase_popularity.get_name()][0]
    assert (v == 38.0 / 10.0)


def test_deep_agg_feat_chain(entityset, backend):
    """
    Agg feat of agg feat:
        region.Mean(customer.Count(Log))
    """
    customer_count_feat = ft.Feature(entityset['log']['id'], parent_entity=entityset['customers'], primitive=Count)

    region_avg_feat = ft.Feature(customer_count_feat, parent_entity=entityset[u'régions'], primitive=Mean)

    pandas_backend = backend([region_avg_feat])
    df = pandas_backend.calculate_all_features(instance_ids=['United States'],
                                               time_last=None)
    v = df[region_avg_feat.get_name()][0]
    assert (v == 17 / 3.)


def test_topn(entityset, backend):
    topn = ft.Feature(entityset['log']['product_id'],
                      parent_entity=entityset['customers'],
                      primitive=NMostCommon(n=2))
    pandas_backend = backend([topn])

    df = pandas_backend.calculate_all_features(instance_ids=[0, 1, 2],
                                               time_last=None)

    true_results = pd.DataFrame([
        ['toothpaste', 'coke zero'],
        ['coke zero', 'Haribo sugar-free gummy bears'],
        ['taco clock', np.nan]
    ])
    assert ([name in df.columns for name in topn.get_feature_names()])
    for i in range(df.shape[0]):
        if i == 0:
            # coke zero and toothpase have same number of occurrences
            assert set(true_results.loc[i].values) == set(df.loc[i].values)
        else:
            for i1, i2 in zip(true_results.loc[i], df.iloc[i]):
                assert (pd.isnull(i1) and pd.isnull(i2)) or (i1 == i2)


def test_trend(entityset, backend):
    trend = ft.Feature([entityset['log']['value'], entityset['log']['datetime']],
                       parent_entity=entityset['customers'],
                       primitive=Trend)
    pandas_backend = backend([trend])

    df = pandas_backend.calculate_all_features(instance_ids=[0, 1, 2],
                                               time_last=None)

    true_results = [-0.812730, 4.870378, np.nan]

    np.testing.assert_almost_equal(df[trend.get_name()].values.tolist(), true_results, decimal=5)


def test_direct_squared(entityset, backend):
    feature = IdentityFeature(entityset['log']['value'])
    squared = feature * feature
    pandas_backend = backend([feature, squared])
    df = pandas_backend.calculate_all_features(instance_ids=[0, 1, 2],
                                               time_last=None)
    for i, row in df.iterrows():
        assert (row[0] * row[0]) == row[1]


def test_agg_empty_child(entityset, backend):
    customer_count_feat = ft.Feature(entityset['log']['id'], parent_entity=entityset['customers'], primitive=Count)
    pandas_backend = backend([customer_count_feat])

    # time last before the customer had any events, so child frame is empty
    df = pandas_backend.calculate_all_features(instance_ids=[0],
                                               time_last=datetime(2011, 4, 8))

    assert df["COUNT(log)"].iloc[0] == 0


def test_empty_child_dataframe():
    parent_df = pd.DataFrame({"id": [1]})
    child_df = pd.DataFrame({"id": [1, 2, 3],
                             "parent_id": [1, 1, 1],
                             "time_index": pd.date_range(start='1/1/2018', periods=3),
                             "value": [10, 5, 2]})

    es = ft.EntitySet(id="blah")
    es.entity_from_dataframe(entity_id="parent", dataframe=parent_df, index="id")
    es.entity_from_dataframe(entity_id="child", dataframe=child_df, index="id", time_index="time_index")
    es.add_relationship(ft.Relationship(es["parent"]["id"], es["child"]["parent_id"]))

    # create regular agg
    count = ft.Feature(es["child"]['id'], parent_entity=es["parent"], primitive=Count)

    # create agg feature that requires multiple arguments
    trend = ft.Feature([es["child"]['value'], es["child"]['time_index']], parent_entity=es["parent"], primitive=Trend)

    # create aggs with where
    where = ft.Feature(es["child"]["value"]) == 1
    count_where = ft.Feature(es["child"]['id'], parent_entity=es["parent"], where=where, primitive=Count)
    trend_where = ft.Feature([es["child"]['value'], es["child"]['time_index']], parent_entity=es["parent"], where=where, primitive=Trend)

    # cutoff time before all rows
    fm = ft.calculate_feature_matrix(entityset=es, features=[count, count_where, trend, trend_where], cutoff_time=pd.Timestamp("12/31/2017"))
    names = [count.get_name(), count_where.get_name(), trend.get_name(), trend_where.get_name()]
    assert_array_equal(fm[names], [[0, 0, np.nan, np.nan]])

    # cutoff time after all rows, but where clause filters all rows
    fm2 = ft.calculate_feature_matrix(entityset=es, features=[count_where, trend_where], cutoff_time=pd.Timestamp("1/4/2018"))
    names = [count_where.get_name(), trend_where.get_name()]
    assert_array_equal(fm2[names], [[0, np.nan]])


def test_handles_primitive_function_name_uniqueness(entityset):
    class SumTimesN(AggregationPrimitive):
        name = "sum_times_n"
        input_types = [Numeric]
        return_type = Numeric

        def __init__(self, n):
            self.n = n

        def get_function(self):
            def my_function(values):
                return values.sum() * self.n

            return my_function

        def generate_name(self, base_feature_names, child_entity_id,
                          parent_entity_id, where_str, use_prev_str):
            base_features_str = ", ".join(base_feature_names)
            return u"%s(%s.%s%s%s, n=%s)" % (self.name.upper(),
                                             child_entity_id,
                                             base_features_str,
                                             where_str, use_prev_str, self.n)

    # works as expected
    f1 = ft.Feature(entityset["log"]["value"],
                    parent_entity=entityset["customers"],
                    primitive=SumTimesN(n=1))
    fm = ft.calculate_feature_matrix(features=[f1], entityset=entityset)
    value_sum = pd.Series([56, 26, 0])
    assert all(fm[f1.get_name()].sort_index() == value_sum)

    # works as expected
    f2 = ft.Feature(entityset["log"]["value"],
                    parent_entity=entityset["customers"],
                    primitive=SumTimesN(n=2))
    fm = ft.calculate_feature_matrix(features=[f2], entityset=entityset)
    double_value_sum = pd.Series([112, 52, 0])
    assert all(fm[f2.get_name()].sort_index() == double_value_sum)

    # same primitive, same variable, different args
    fm = ft.calculate_feature_matrix(features=[f1, f2], entityset=entityset)
    assert all(fm[f1.get_name()].sort_index() == value_sum)
    assert all(fm[f2.get_name()].sort_index() == double_value_sum)

    # different primtives, same function returned by get_function,
    # different base features
    f3 = ft.Feature(entityset["log"]["value"],
                    parent_entity=entityset["customers"],
                    primitive=Sum)
    f4 = ft.Feature(entityset["log"]["purchased"],
                    parent_entity=entityset["customers"],
                    primitive=NumTrue)
    fm = ft.calculate_feature_matrix(features=[f3, f4], entityset=entityset)
    purchased_sum = pd.Series([10, 1, 1])
    assert all(fm[f3.get_name()].sort_index() == value_sum)
    assert all(fm[f4.get_name()].sort_index() == purchased_sum)\


    # different primtives, same function returned by get_function,
    # same base feature
    class Sum1(AggregationPrimitive):
        """Sums elements of a numeric or boolean feature."""
        name = "sum1"
        input_types = [Numeric]
        return_type = Numeric
        stack_on_self = False
        stack_on_exclude = [Count]
        default_value = 0

        def get_function(self):
            return np.sum

    class Sum2(AggregationPrimitive):
        """Sums elements of a numeric or boolean feature."""
        name = "sum2"
        input_types = [Numeric]
        return_type = Numeric
        stack_on_self = False
        stack_on_exclude = [Count]
        default_value = 0

        def get_function(self):
            return np.sum

    class Sum3(AggregationPrimitive):
        """Sums elements of a numeric or boolean feature."""
        name = "sum3"
        input_types = [Numeric]
        return_type = Numeric
        stack_on_self = False
        stack_on_exclude = [Count]
        default_value = 0

        def get_function(self):
            return np.sum

    f5 = ft.Feature(entityset["log"]["value"],
                    parent_entity=entityset["customers"],
                    primitive=Sum1)
    f6 = ft.Feature(entityset["log"]["value"],
                    parent_entity=entityset["customers"],
                    primitive=Sum2)
    f7 = ft.Feature(entityset["log"]["value"],
                    parent_entity=entityset["customers"],
                    primitive=Sum3)
    fm = ft.calculate_feature_matrix(features=[f5, f6, f7], entityset=entityset)
    assert all(fm[f5.get_name()].sort_index() == value_sum)
    assert all(fm[f6.get_name()].sort_index() == value_sum)
    assert all(fm[f7.get_name()].sort_index() == value_sum)
