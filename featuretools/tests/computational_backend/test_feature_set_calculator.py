import sys
from datetime import datetime

import numpy as np
import pandas as pd
import pytest
from dask import dataframe as dd
from numpy.testing import assert_array_equal

import featuretools as ft
from featuretools import Timedelta, variable_types
from featuretools.computational_backends.feature_set import FeatureSet
from featuretools.computational_backends.feature_set_calculator import (
    FeatureSetCalculator
)
from featuretools.entityset.relationship import RelationshipPath
from featuretools.feature_base import DirectFeature, IdentityFeature
from featuretools.primitives import (  # NMostCommon,
    And,
    Count,
    CumSum,
    EqualScalar,
    GreaterThanEqualToScalar,
    GreaterThanScalar,
    LessThanEqualToScalar,
    LessThanScalar,
    Mean,
    Min,
    Mode,
    Negate,
    NMostCommon,
    NotEqualScalar,
    NumTrue,
    Sum,
    TimeSinceLast,
    Trend
)
from featuretools.primitives.base import AggregationPrimitive
from featuretools.tests.testing_utils import backward_path, to_pandas
from featuretools.utils import Trie
from featuretools.variable_types import Numeric


def test_make_identity(es):
    f = IdentityFeature(es['log']['datetime'])

    feature_set = FeatureSet([f])
    calculator = FeatureSetCalculator(es,
                                      time_last=None,
                                      feature_set=feature_set)
    df = to_pandas(calculator.run(np.array([0])))

    v = df[f.get_name()][0]
    assert (v == datetime(2011, 4, 9, 10, 30, 0))


def test_make_dfeat(es):
    f = DirectFeature(es['customers']['age'],
                      child_entity=es['sessions'])

    feature_set = FeatureSet([f])
    calculator = FeatureSetCalculator(es,
                                      time_last=None,
                                      feature_set=feature_set)
    df = to_pandas(calculator.run(np.array([0])))

    v = df[f.get_name()][0]
    assert (v == 33)


def test_make_agg_feat_of_identity_variable(es):
    agg_feat = ft.Feature(es['log']['value'], parent_entity=es['sessions'], primitive=Sum)

    feature_set = FeatureSet([agg_feat])
    calculator = FeatureSetCalculator(es,
                                      time_last=None,
                                      feature_set=feature_set)
    df = to_pandas(calculator.run(np.array([0])))

    v = df[agg_feat.get_name()][0]
    assert (v == 50)


# full_entity not supported with Dask
def test_full_entity_trans_of_agg(pd_es):
    agg_feat = ft.Feature(pd_es['log']['value'], parent_entity=pd_es['customers'],
                          primitive=Sum)
    trans_feat = ft.Feature(agg_feat, primitive=CumSum)

    feature_set = FeatureSet([trans_feat])
    calculator = FeatureSetCalculator(pd_es,
                                      time_last=None,
                                      feature_set=feature_set)
    df = calculator.run(np.array([1]))

    v = df[trans_feat.get_name()].values[0]
    assert v == 82


def test_full_entity_error_dask(dask_es):
    agg_feat = ft.Feature(dask_es['log']['value'], parent_entity=dask_es['customers'],
                          primitive=Sum)
    trans_feat = ft.Feature(agg_feat, primitive=CumSum)

    feature_set = FeatureSet([trans_feat])
    calculator = FeatureSetCalculator(dask_es,
                                      time_last=None,
                                      feature_set=feature_set)
    error_text = "Cannot use primitives that require full entity with Dask"

    with pytest.raises(ValueError, match=error_text):
        calculator.run(np.array([1]))


def test_make_agg_feat_of_identity_index_variable(es):
    agg_feat = ft.Feature(es['log']['id'], parent_entity=es['sessions'], primitive=Count)

    feature_set = FeatureSet([agg_feat])
    calculator = FeatureSetCalculator(es,
                                      time_last=None,
                                      feature_set=feature_set)
    df = to_pandas(calculator.run(np.array([0])))

    v = df[agg_feat.get_name()][0]
    assert (v == 5)


def test_make_agg_feat_where_count(es):
    agg_feat = ft.Feature(es['log']['id'],
                          parent_entity=es['sessions'],
                          where=IdentityFeature(es['log']['product_id']) == 'coke zero',
                          primitive=Count)

    feature_set = FeatureSet([agg_feat])
    calculator = FeatureSetCalculator(es,
                                      time_last=None,
                                      feature_set=feature_set)
    df = to_pandas(calculator.run(np.array([0])))

    v = df[agg_feat.get_name()][0]
    assert (v == 3)


def test_make_agg_feat_using_prev_time(es):
    agg_feat = ft.Feature(es['log']['id'],
                          parent_entity=es['sessions'],
                          use_previous=Timedelta(10, 's'),
                          primitive=Count)

    feature_set = FeatureSet([agg_feat])
    calculator = FeatureSetCalculator(es,
                                      time_last=datetime(2011, 4, 9, 10, 30, 10),
                                      feature_set=feature_set)
    df = to_pandas(calculator.run(np.array([0])))

    v = df[agg_feat.get_name()][0]
    assert (v == 2)

    calculator = FeatureSetCalculator(es,
                                      time_last=datetime(2011, 4, 9, 10, 30, 30),
                                      feature_set=feature_set)
    df = to_pandas(calculator.run(np.array([0])))

    v = df[agg_feat.get_name()][0]
    assert (v == 1)


def test_make_agg_feat_using_prev_n_events(es):
    if not all(isinstance(entity.df, pd.DataFrame) for entity in es.entities):
        pytest.xfail('Distrubuted entitysets do not support use_previous')
    agg_feat_1 = ft.Feature(es['log']['value'],
                            parent_entity=es['sessions'],
                            use_previous=Timedelta(1, 'observations'),
                            primitive=Min)

    agg_feat_2 = ft.Feature(es['log']['value'],
                            parent_entity=es['sessions'],
                            use_previous=Timedelta(3, 'observations'),
                            primitive=Min)

    assert agg_feat_1.get_name() != agg_feat_2.get_name(), \
        'Features should have different names based on use_previous'

    feature_set = FeatureSet([agg_feat_1, agg_feat_2])
    calculator = FeatureSetCalculator(es,
                                      time_last=datetime(2011, 4, 9, 10, 30, 6),
                                      feature_set=feature_set)
    df = calculator.run(np.array([0]))

    # time_last is included by default
    v1 = df[agg_feat_1.get_name()][0]
    v2 = df[agg_feat_2.get_name()][0]
    assert v1 == 5
    assert v2 == 0

    calculator = FeatureSetCalculator(es,
                                      time_last=datetime(2011, 4, 9, 10, 30, 30),
                                      feature_set=feature_set)
    df = calculator.run(np.array([0]))

    v1 = df[agg_feat_1.get_name()][0]
    v2 = df[agg_feat_2.get_name()][0]
    assert v1 == 20
    assert v2 == 10


def test_make_agg_feat_multiple_dtypes(es):
    if not all(isinstance(entity.df, pd.DataFrame) for entity in es.entities):
        pytest.xfail('Currently no Dask or Koalas compatible agg prims that use multiple dtypes')
    compare_prod = IdentityFeature(es['log']['product_id']) == 'coke zero'

    agg_feat = ft.Feature(es['log']['id'],
                          parent_entity=es['sessions'],
                          where=compare_prod,
                          primitive=Count)

    agg_feat2 = ft.Feature(es['log']['product_id'],
                           parent_entity=es['sessions'],
                           where=compare_prod,
                           primitive=Mode)

    feature_set = FeatureSet([agg_feat, agg_feat2])
    calculator = FeatureSetCalculator(es,
                                      time_last=None,
                                      feature_set=feature_set)
    df = calculator.run(np.array([0]))

    v = df[agg_feat.get_name()][0]
    v2 = df[agg_feat2.get_name()][0]
    assert (v == 3)
    assert (v2 == 'coke zero')


def test_make_agg_feat_where_different_identity_feat(es):
    feats = []
    where_cmps = [LessThanScalar, GreaterThanScalar, LessThanEqualToScalar,
                  GreaterThanEqualToScalar, EqualScalar, NotEqualScalar]
    for where_cmp in where_cmps:
        feats.append(ft.Feature(es['log']['id'],
                                parent_entity=es['sessions'],
                                where=ft.Feature(es['log']['datetime'], primitive=where_cmp(datetime(2011, 4, 10, 10, 40, 1))),
                                primitive=Count))

    df = ft.calculate_feature_matrix(entityset=es, features=feats, instance_ids=[0, 1, 2, 3])
    df = to_pandas(df, index='id', sort_index=True)

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


def test_make_agg_feat_of_grandchild_entity(es):
    agg_feat = ft.Feature(es['log']['id'], parent_entity=es['customers'], primitive=Count)

    feature_set = FeatureSet([agg_feat])
    calculator = FeatureSetCalculator(es,
                                      time_last=None,
                                      feature_set=feature_set)
    df = calculator.run(np.array([0]))
    df = to_pandas(df, index='id')
    v = df[agg_feat.get_name()].values[0]
    assert (v == 10)


def test_make_agg_feat_where_count_feat(es):
    """
    Feature we're creating is:
    Number of sessions for each customer where the
    number of logs in the session is less than 3
    """
    log_count_feat = ft.Feature(es['log']['id'], parent_entity=es['sessions'], primitive=Count)

    feat = ft.Feature(es['sessions']['id'],
                      parent_entity=es['customers'],
                      where=log_count_feat > 1,
                      primitive=Count)

    feature_set = FeatureSet([feat])
    calculator = FeatureSetCalculator(es,
                                      time_last=None,
                                      feature_set=feature_set)
    df = calculator.run(np.array([0, 1]))
    df = to_pandas(df, index='id', sort_index=True)

    name = feat.get_name()
    instances = df[name]
    v0, v1 = instances[0:2]
    assert (v0 == 2)
    assert (v1 == 2)


def test_make_compare_feat(es):
    """
    Feature we're creating is:
    Number of sessions for each customer where the
    number of logs in the session is less than 3
    """
    log_count_feat = ft.Feature(es['log']['id'], parent_entity=es['sessions'], primitive=Count)

    mean_agg_feat = ft.Feature(log_count_feat, parent_entity=es['customers'], primitive=Mean)

    mean_feat = DirectFeature(mean_agg_feat, child_entity=es['sessions'])

    feat = log_count_feat > mean_feat

    feature_set = FeatureSet([feat])
    calculator = FeatureSetCalculator(es,
                                      time_last=None,
                                      feature_set=feature_set)
    df = calculator.run(np.array([0, 1, 2]))
    df = to_pandas(df, index='id', sort_index=True)

    name = feat.get_name()
    instances = df[name]
    v0, v1, v2 = instances[0:3]
    assert v0
    assert v1
    assert not v2


def test_make_agg_feat_where_count_and_device_type_feat(es):
    """
    Feature we're creating is:
    Number of sessions for each customer where the
    number of logs in the session is less than 3
    """
    log_count_feat = ft.Feature(es['log']['id'], parent_entity=es['sessions'], primitive=Count)

    compare_count = log_count_feat == 1
    compare_device_type = IdentityFeature(es['sessions']['device_type']) == 1
    and_feat = ft.Feature([compare_count, compare_device_type], primitive=And)
    feat = ft.Feature(es['sessions']['id'],
                      parent_entity=es['customers'],
                      where=and_feat,
                      primitive=Count)

    feature_set = FeatureSet([feat])
    calculator = FeatureSetCalculator(es,
                                      time_last=None,
                                      feature_set=feature_set)
    df = calculator.run(np.array([0]))
    df = to_pandas(df, index='id')

    name = feat.get_name()
    instances = df[name]
    assert (instances.values[0] == 1)


def test_make_agg_feat_where_count_or_device_type_feat(es):
    """
    Feature we're creating is:
    Number of sessions for each customer where the
    number of logs in the session is less than 3
    """
    log_count_feat = ft.Feature(es['log']['id'], parent_entity=es['sessions'], primitive=Count)

    compare_count = log_count_feat > 1
    compare_device_type = IdentityFeature(es['sessions']['device_type']) == 1
    or_feat = compare_count.OR(compare_device_type)
    feat = ft.Feature(es['sessions']['id'],
                      parent_entity=es['customers'],
                      where=or_feat,
                      primitive=Count)

    feature_set = FeatureSet([feat])
    calculator = FeatureSetCalculator(es,
                                      time_last=None,
                                      feature_set=feature_set)
    df = calculator.run(np.array([0]))
    df = to_pandas(df, index='id', int_index=True)

    name = feat.get_name()
    instances = df[name]
    assert (instances.values[0] == 3)


def test_make_agg_feat_of_agg_feat(es):
    log_count_feat = ft.Feature(es['log']['id'], parent_entity=es['sessions'], primitive=Count)

    customer_sum_feat = ft.Feature(log_count_feat, parent_entity=es['customers'], primitive=Sum)

    feature_set = FeatureSet([customer_sum_feat])
    calculator = FeatureSetCalculator(es,
                                      time_last=None,
                                      feature_set=feature_set)
    df = calculator.run(np.array([0]))
    df = to_pandas(df, index='id')
    v = df[customer_sum_feat.get_name()].values[0]
    assert (v == 10)


@pytest.fixture
def pd_df():
    return pd.DataFrame({
        "id": ["a", "b", "c", "d", "e"],
        "e1": ["h", "h", "i", "i", "j"],
        "e2": ["x", "x", "y", "y", "x"],
        "e3": ["z", "z", "z", "z", "z"],
        "val": [1, 1, 1, 1, 1]
    })


@pytest.fixture
def dd_df(pd_df):
    return dd.from_pandas(pd_df, npartitions=2)


@pytest.fixture
def ks_df(pd_df):
    ks = pytest.importorskip('databricks.koalas', reason="Koalas not installed, skipping")
    if sys.platform.startswith('win'):
        pytest.skip('skipping Koalas tests for Windows')
    return ks.from_pandas(pd_df)


@pytest.fixture(params=['pd_df', 'dd_df', 'ks_df'])
def df(request):
    return request.getfixturevalue(request.param)


def test_make_3_stacked_agg_feats(df):
    """
    Tests stacking 3 agg features.

    The test specifically uses non numeric indices to test how ancestor variables are handled
    as dataframes are merged together

    """
    if isinstance(df, dd.DataFrame):
        pytest.xfail('normalize_entity fails with dask DataFrame')
    es = ft.EntitySet()
    vtypes = {
        'id': variable_types.Index,
        'e1': variable_types.Categorical,
        'e2': variable_types.Categorical,
        'e3': variable_types.Categorical,
        'val': variable_types.Numeric
    }
    es.entity_from_dataframe(dataframe=df,
                             index="id",
                             entity_id="e0",
                             variable_types=vtypes)

    es.normalize_entity(base_entity_id="e0",
                        new_entity_id="e1",
                        index="e1",
                        additional_variables=["e2", "e3"])

    es.normalize_entity(base_entity_id="e1",
                        new_entity_id="e2",
                        index="e2",
                        additional_variables=["e3"])

    es.normalize_entity(base_entity_id="e2",
                        new_entity_id="e3",
                        index="e3")

    sum_1 = ft.Feature(es["e0"]["val"], parent_entity=es["e1"], primitive=Sum)
    sum_2 = ft.Feature(sum_1, parent_entity=es["e2"], primitive=Sum)
    sum_3 = ft.Feature(sum_2, parent_entity=es["e3"], primitive=Sum)

    feature_set = FeatureSet([sum_3])
    calculator = FeatureSetCalculator(es,
                                      time_last=None,
                                      feature_set=feature_set)
    df = calculator.run(np.array(["z"]))
    v = df[sum_3.get_name()][0]
    assert (v == 5)


def test_make_dfeat_of_agg_feat_on_self(es):
    """
    The graph looks like this:

        R       R = Regions, a parent of customers
        |
        C       C = Customers, the entity we're trying to predict on
        |
       etc.

    We're trying to calculate a DFeat from C to R on an agg_feat of R on C.
    """
    customer_count_feat = ft.Feature(es['customers']['id'], parent_entity=es[u'régions'], primitive=Count)

    num_customers_feat = DirectFeature(customer_count_feat, child_entity=es['customers'])

    feature_set = FeatureSet([num_customers_feat])
    calculator = FeatureSetCalculator(es,
                                      time_last=None,
                                      feature_set=feature_set)
    df = calculator.run(np.array([0]))
    df = to_pandas(df, index='id')
    v = df[num_customers_feat.get_name()].values[0]
    assert (v == 3)


def test_make_dfeat_of_agg_feat_through_parent(es):
    """
    The graph looks like this:

        R       C = Customers, the entity we're trying to predict on
       / \\     R = Regions, a parent of customers
      S   C     S = Stores, a child of regions
          |
         etc.

    We're trying to calculate a DFeat from C to R on an agg_feat of R on S.
    """
    store_id_feat = IdentityFeature(es['stores']['id'])

    store_count_feat = ft.Feature(store_id_feat, parent_entity=es[u'régions'], primitive=Count)

    num_stores_feat = DirectFeature(store_count_feat, child_entity=es['customers'])

    feature_set = FeatureSet([num_stores_feat])
    calculator = FeatureSetCalculator(es,
                                      time_last=None,
                                      feature_set=feature_set)
    df = calculator.run(np.array([0]))
    df = to_pandas(df, index='id')
    v = df[num_stores_feat.get_name()].values[0]
    assert (v == 3)


def test_make_deep_agg_feat_of_dfeat_of_agg_feat(es):
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
    log_count_feat = ft.Feature(es['log']['id'], parent_entity=es['products'], primitive=Count)

    product_purchases_feat = DirectFeature(log_count_feat,
                                           child_entity=es['log'])

    purchase_popularity = ft.Feature(product_purchases_feat, parent_entity=es['customers'], primitive=Mean)

    feature_set = FeatureSet([purchase_popularity])
    calculator = FeatureSetCalculator(es,
                                      time_last=None,
                                      feature_set=feature_set)
    df = calculator.run(np.array([0]))
    df = to_pandas(df, index='id')
    v = df[purchase_popularity.get_name()].values[0]
    assert (v == 38.0 / 10.0)


def test_deep_agg_feat_chain(es):
    """
    Agg feat of agg feat:
        region.Mean(customer.Count(Log))
    """
    customer_count_feat = ft.Feature(es['log']['id'], parent_entity=es['customers'], primitive=Count)

    region_avg_feat = ft.Feature(customer_count_feat, parent_entity=es[u'régions'], primitive=Mean)

    feature_set = FeatureSet([region_avg_feat])
    calculator = FeatureSetCalculator(es,
                                      time_last=None,
                                      feature_set=feature_set)
    df = calculator.run(np.array(['United States']))
    df = to_pandas(df, index='id')

    v = df[region_avg_feat.get_name()][0]
    assert (v == 17 / 3.)


# NMostCommon not supported with Dask or Koalas
def test_topn(pd_es):
    topn = ft.Feature(pd_es['log']['product_id'],
                      parent_entity=pd_es['customers'],
                      primitive=NMostCommon(n=2))
    feature_set = FeatureSet([topn])

    calculator = FeatureSetCalculator(pd_es,
                                      time_last=None,
                                      feature_set=feature_set)
    df = calculator.run(np.array([0, 1, 2]))
    true_results = pd.DataFrame([
        ['toothpaste', 'coke zero'],
        ['coke zero', 'Haribo sugar-free gummy bears'],
        ['taco clock', np.nan]
    ])
    assert ([name in df.columns for name in topn.get_feature_names()])

    for i in range(df.shape[0]):
        true = true_results.loc[i]
        actual = df.loc[i]
        if i == 0:
            # coke zero and toothpase have same number of occurrences
            assert set(true.values) == set(actual.values)
        else:
            for i1, i2 in zip(true, actual):
                assert (pd.isnull(i1) and pd.isnull(i2)) or (i1 == i2)


# Trend not supported with Dask or Koalas
def test_trend(pd_es):
    trend = ft.Feature([pd_es['log']['value'], pd_es['log']['datetime']],
                       parent_entity=pd_es['customers'],
                       primitive=Trend)
    feature_set = FeatureSet([trend])

    calculator = FeatureSetCalculator(pd_es,
                                      time_last=None,
                                      feature_set=feature_set)
    df = calculator.run(np.array([0, 1, 2]))

    true_results = [-0.812730, 4.870378, np.nan]

    np.testing.assert_almost_equal(df[trend.get_name()].values.tolist(), true_results, decimal=5)


def test_direct_squared(es):
    feature = IdentityFeature(es['log']['value'])
    squared = feature * feature
    feature_set = FeatureSet([feature, squared])
    calculator = FeatureSetCalculator(es,
                                      time_last=None,
                                      feature_set=feature_set)
    df = to_pandas(calculator.run(np.array([0, 1, 2])))
    for i, row in df.iterrows():
        assert (row[0] * row[0]) == row[1]


def test_agg_empty_child(es):
    customer_count_feat = ft.Feature(es['log']['id'], parent_entity=es['customers'], primitive=Count)
    feature_set = FeatureSet([customer_count_feat])

    # time last before the customer had any events, so child frame is empty
    calculator = FeatureSetCalculator(es,
                                      time_last=datetime(2011, 4, 8),
                                      feature_set=feature_set)
    df = to_pandas(calculator.run(np.array([0])), index='id')

    assert df["COUNT(log)"].iloc[0] == 0


def test_diamond_entityset(diamond_es):
    es = diamond_es

    amount = ft.IdentityFeature(es['transactions']['amount'])
    path = backward_path(es, ['regions', 'customers', 'transactions'])
    through_customers = ft.AggregationFeature(amount, es['regions'],
                                              primitive=ft.primitives.Sum,
                                              relationship_path=path)
    path = backward_path(es, ['regions', 'stores', 'transactions'])
    through_stores = ft.AggregationFeature(amount, es['regions'],
                                           primitive=ft.primitives.Sum,
                                           relationship_path=path)

    feature_set = FeatureSet([through_customers, through_stores])
    calculator = FeatureSetCalculator(es,
                                      time_last=datetime(2011, 4, 8),
                                      feature_set=feature_set)
    df = calculator.run(np.array([0, 1, 2]))
    df = to_pandas(df, index='id', sort_index=True)

    assert (df['SUM(stores.transactions.amount)'] == [94, 261, 128]).all()
    assert (df['SUM(customers.transactions.amount)'] == [72, 411, 0]).all()


def test_two_relationships_to_single_entity(games_es):
    es = games_es
    home_team, away_team = es.relationships
    path = RelationshipPath([(False, home_team)])
    mean_at_home = ft.AggregationFeature(es['games']['home_team_score'],
                                         es['teams'],
                                         relationship_path=path,
                                         primitive=ft.primitives.Mean)
    path = RelationshipPath([(False, away_team)])
    mean_at_away = ft.AggregationFeature(es['games']['away_team_score'],
                                         es['teams'],
                                         relationship_path=path,
                                         primitive=ft.primitives.Mean)
    home_team_mean = ft.DirectFeature(mean_at_home, es['games'],
                                      relationship=home_team)
    away_team_mean = ft.DirectFeature(mean_at_away, es['games'],
                                      relationship=away_team)

    feature_set = FeatureSet([home_team_mean, away_team_mean])
    calculator = FeatureSetCalculator(es,
                                      time_last=datetime(2011, 8, 28),
                                      feature_set=feature_set)
    df = calculator.run(np.array(range(3)))
    df = to_pandas(df, index='id', sort_index=True)

    assert (df[home_team_mean.get_name()] == [1.5, 1.5, 2.5]).all()
    assert (df[away_team_mean.get_name()] == [1, 0.5, 2]).all()


@pytest.fixture
def pd_parent_child():
    parent_df = pd.DataFrame({"id": [1]})
    child_df = pd.DataFrame({"id": [1, 2, 3],
                             "parent_id": [1, 1, 1],
                             "time_index": pd.date_range(start='1/1/2018', periods=3),
                             "value": [10, 5, 2],
                             "cat": ['a', 'a', 'b']})
    return (parent_df, child_df)


@pytest.fixture
def dd_parent_child(pd_parent_child):
    parent_df, child_df = pd_parent_child
    parent_df = dd.from_pandas(parent_df, npartitions=2)
    child_df = dd.from_pandas(child_df, npartitions=2)
    return (parent_df, child_df)


@pytest.fixture
def ks_parent_child(pd_parent_child):
    ks = pytest.importorskip('databricks.koalas', reason="Koalas not installed, skipping")
    if sys.platform.startswith('win'):
        pytest.skip('skipping Koalas tests for Windows')
    parent_df, child_df = pd_parent_child
    parent_df = ks.from_pandas(parent_df)
    child_df = ks.from_pandas(child_df)
    return (parent_df, child_df)


@pytest.fixture(params=['pd_parent_child', 'dd_parent_child', 'ks_parent_child'])
def parent_child(request):
    return request.getfixturevalue(request.param)


def test_empty_child_dataframe(parent_child):
    parent_df, child_df = parent_child
    if not isinstance(parent_df, pd.DataFrame):
        parent_vtypes = {
            'id': variable_types.Index
        }
        child_vtypes = {
            'id': variable_types.Index,
            'parent_id': variable_types.Numeric,
            'time_index': variable_types.Datetime,
            'value': variable_types.Numeric,
            'cat': variable_types.Categorical
        }
    else:
        parent_vtypes = None
        child_vtypes = None
    es = ft.EntitySet(id="blah")
    es.entity_from_dataframe(entity_id="parent",
                             dataframe=parent_df,
                             index="id",
                             variable_types=parent_vtypes)
    es.entity_from_dataframe(entity_id="child",
                             dataframe=child_df,
                             index="id",
                             time_index="time_index",
                             variable_types=child_vtypes)
    es.add_relationship(ft.Relationship(es["parent"]["id"], es["child"]["parent_id"]))

    # create regular agg
    count = ft.Feature(es["child"]['id'], parent_entity=es["parent"], primitive=Count)

    # create agg feature that requires multiple arguments
    trend = ft.Feature([es["child"]['value'], es["child"]['time_index']], parent_entity=es["parent"], primitive=Trend)

    # create multi-output agg feature
    n_most_common = ft.Feature(es["child"]['cat'], parent_entity=es["parent"], primitive=NMostCommon)

    # create aggs with where
    where = ft.Feature(es["child"]["value"]) == 1
    count_where = ft.Feature(es["child"]['id'], parent_entity=es["parent"], where=where, primitive=Count)
    trend_where = ft.Feature([es["child"]['value'], es["child"]['time_index']], parent_entity=es["parent"], where=where, primitive=Trend)
    n_most_common_where = ft.Feature(es["child"]['cat'], parent_entity=es["parent"], where=where, primitive=NMostCommon)

    if isinstance(parent_df, pd.DataFrame):
        features = [count, count_where, trend, trend_where, n_most_common, n_most_common_where]
        names = [count.get_name(), count_where.get_name(),
                 trend.get_name(), trend_where.get_name(),
                 *n_most_common.get_feature_names(), *n_most_common_where.get_feature_names()]
        values = [0, 0,
                  np.nan, np.nan,
                  *np.full(n_most_common.number_output_features, np.nan), *np.full(n_most_common_where.number_output_features, np.nan)]
    else:
        features = [count, count_where]
        names = [count.get_name(), count_where.get_name()]
        values = [0, 0]

    # cutoff time before all rows
    fm = ft.calculate_feature_matrix(entityset=es,
                                     features=features,
                                     cutoff_time=pd.Timestamp("12/31/2017"))
    fm = to_pandas(fm)

    assert_array_equal(fm[names], [values])

    # cutoff time after all rows, but where clause filters all rows
    if isinstance(parent_df, pd.DataFrame):
        features = [count_where, trend_where, n_most_common_where]
        names = [count_where.get_name(), trend_where.get_name(), *n_most_common_where.get_feature_names()]
        values = [0, np.nan, *np.full(n_most_common_where.number_output_features, np.nan)]
    else:
        features = [count_where]
        names = [count_where.get_name()]
        values = [0]

    fm2 = ft.calculate_feature_matrix(entityset=es,
                                      features=features,
                                      cutoff_time=pd.Timestamp("1/4/2018"))
    fm2 = to_pandas(fm2)

    assert_array_equal(fm2[names], [values])


def test_with_features_built_from_es_metadata(es):
    metadata = es.metadata
    agg_feat = ft.Feature(metadata['log']['id'], parent_entity=metadata['customers'], primitive=Count)

    feature_set = FeatureSet([agg_feat])
    calculator = FeatureSetCalculator(es,
                                      time_last=None,
                                      feature_set=feature_set)
    df = calculator.run(np.array([0]))
    df = to_pandas(df, index='id')
    v = df[agg_feat.get_name()].values[0]
    assert (v == 10)


# TODO: Fails with Dask and Koalas (conflicting aggregation primitives)
def test_handles_primitive_function_name_uniqueness(es):
    if not all(isinstance(entity.df, pd.DataFrame) for entity in es.entities):
        pytest.xfail("Fails with Dask and Koalas due conflicting aggregation primitive names")

    class SumTimesN(AggregationPrimitive):
        name = "sum_times_n"
        input_types = [Numeric]
        return_type = Numeric

        def __init__(self, n):
            self.n = n

        def get_function(self, agg_type='pandas'):
            def my_function(values):
                return values.sum() * self.n

            return my_function

    # works as expected
    f1 = ft.Feature(es["log"]["value"],
                    parent_entity=es["customers"],
                    primitive=SumTimesN(n=1))
    fm = ft.calculate_feature_matrix(features=[f1], entityset=es)

    value_sum = pd.Series([56, 26, 0])
    assert all(fm[f1.get_name()].sort_index() == value_sum)

    # works as expected
    f2 = ft.Feature(es["log"]["value"],
                    parent_entity=es["customers"],
                    primitive=SumTimesN(n=2))
    fm = ft.calculate_feature_matrix(features=[f2], entityset=es)

    double_value_sum = pd.Series([112, 52, 0])
    assert all(fm[f2.get_name()].sort_index() == double_value_sum)

    # same primitive, same variable, different args
    fm = ft.calculate_feature_matrix(features=[f1, f2], entityset=es)

    assert all(fm[f1.get_name()].sort_index() == value_sum)
    assert all(fm[f2.get_name()].sort_index() == double_value_sum)

    # different primtives, same function returned by get_function,
    # different base features
    f3 = ft.Feature(es["log"]["value"],
                    parent_entity=es["customers"],
                    primitive=Sum)
    f4 = ft.Feature(es["log"]["purchased"],
                    parent_entity=es["customers"],
                    primitive=NumTrue)
    fm = ft.calculate_feature_matrix(features=[f3, f4], entityset=es)

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

        def get_function(self, agg_type='pandas'):
            return np.sum

    class Sum2(AggregationPrimitive):
        """Sums elements of a numeric or boolean feature."""
        name = "sum2"
        input_types = [Numeric]
        return_type = Numeric
        stack_on_self = False
        stack_on_exclude = [Count]
        default_value = 0

        def get_function(self, agg_type='pandas'):
            return np.sum

    class Sum3(AggregationPrimitive):
        """Sums elements of a numeric or boolean feature."""
        name = "sum3"
        input_types = [Numeric]
        return_type = Numeric
        stack_on_self = False
        stack_on_exclude = [Count]
        default_value = 0

        def get_function(self, agg_type='pandas'):
            return np.sum

    f5 = ft.Feature(es["log"]["value"],
                    parent_entity=es["customers"],
                    primitive=Sum1)
    f6 = ft.Feature(es["log"]["value"],
                    parent_entity=es["customers"],
                    primitive=Sum2)
    f7 = ft.Feature(es["log"]["value"],
                    parent_entity=es["customers"],
                    primitive=Sum3)
    fm = ft.calculate_feature_matrix(features=[f5, f6, f7], entityset=es)
    assert all(fm[f5.get_name()].sort_index() == value_sum)
    assert all(fm[f6.get_name()].sort_index() == value_sum)
    assert all(fm[f7.get_name()].sort_index() == value_sum)


# No order guarantees w/ Dask
def test_returns_order_of_instance_ids(pd_es):
    feature_set = FeatureSet([ft.Feature(pd_es['customers']['age'])])
    calculator = FeatureSetCalculator(pd_es,
                                      time_last=None,
                                      feature_set=feature_set)

    instance_ids = [0, 1, 2]
    assert list(pd_es['customers'].df['id']) != instance_ids

    df = calculator.run(np.array(instance_ids))

    assert list(df.index) == instance_ids


def test_calls_progress_callback(es):
    # call with all feature types. make sure progress callback calls sum to 1
    identity = ft.Feature(es['customers']['age'])
    direct = ft.Feature(es['cohorts']['cohort_name'], es['customers'])
    agg = ft.Feature(es["sessions"]["id"], parent_entity=es['customers'], primitive=Count)
    agg_apply = ft.Feature(es["log"]["datetime"], parent_entity=es['customers'], primitive=TimeSinceLast)  # this feature is handle differently than simple features
    trans = ft.Feature(agg, primitive=Negate)
    trans_full = ft.Feature(agg, primitive=CumSum)
    groupby_trans = ft.Feature(agg, primitive=CumSum, groupby=es["customers"]["cohort"])

    if not all(isinstance(entity.df, pd.DataFrame) for entity in es.entities):
        all_features = [identity, direct, agg, trans]
    else:
        all_features = [identity, direct, agg, agg_apply, trans, trans_full, groupby_trans]

    feature_set = FeatureSet(all_features)
    calculator = FeatureSetCalculator(es,
                                      time_last=None,
                                      feature_set=feature_set)

    class MockProgressCallback:
        def __init__(self):
            self.total = 0

        def __call__(self, update):
            self.total += update

    mock_progress_callback = MockProgressCallback()

    instance_ids = [0, 1, 2]
    calculator.run(np.array(instance_ids), mock_progress_callback)

    assert np.isclose(mock_progress_callback.total, 1)

    # testing again with a time_last with no data
    feature_set = FeatureSet(all_features)
    calculator = FeatureSetCalculator(es,
                                      time_last=pd.Timestamp("1950"),
                                      feature_set=feature_set)

    mock_progress_callback = MockProgressCallback()
    calculator.run(np.array(instance_ids), mock_progress_callback)

    assert np.isclose(mock_progress_callback.total, 1)


# precalcualted_features is only used with approximate
def test_precalculated_features(pd_es):
    error_msg = 'This primitive should never be used because the features are precalculated'

    class ErrorPrim(AggregationPrimitive):
        """A primitive whose function raises an error."""
        name = "error_prim"
        input_types = [Numeric]
        return_type = Numeric

        def get_function(self, agg_type='pandas'):
            def error(s):
                raise RuntimeError(error_msg)
            return error

    value = ft.Feature(pd_es['log']['value'])
    agg = ft.Feature(value,
                     parent_entity=pd_es['sessions'],
                     primitive=ErrorPrim)
    agg2 = ft.Feature(agg,
                      parent_entity=pd_es['customers'],
                      primitive=ErrorPrim)
    direct = ft.Feature(agg2, entity=pd_es['sessions'])

    # Set up a FeatureSet which knows which features are precalculated.
    precalculated_feature_trie = Trie(default=set, path_constructor=RelationshipPath)
    precalculated_feature_trie.get_node(direct.relationship_path).value.add(agg2.unique_name())
    feature_set = FeatureSet([direct], approximate_feature_trie=precalculated_feature_trie)

    # Fake precalculated data.
    values = [0, 1, 2]
    parent_fm = pd.DataFrame({agg2.get_name(): values})
    precalculated_fm_trie = Trie(path_constructor=RelationshipPath)
    precalculated_fm_trie.get_node(direct.relationship_path).value = parent_fm

    calculator = FeatureSetCalculator(pd_es,
                                      feature_set=feature_set,
                                      precalculated_features=precalculated_fm_trie)

    instance_ids = [0, 2, 3, 5]
    fm = calculator.run(np.array(instance_ids))

    assert list(fm[direct.get_name()]) == [values[0], values[0], values[1], values[2]]

    # Calculating without precalculated features should error.
    with pytest.raises(RuntimeError, match=error_msg):
        FeatureSetCalculator(pd_es, feature_set=FeatureSet([direct])).run(instance_ids)
