# -*- coding: utf-8 -*-

import pandas as pd
import pytest
from distributed.utils_test import cluster

from featuretools.entityset import EntitySet, Relationship
from featuretools.primitives import Max, Mean, Min, Sum
from featuretools.synthesis import dfs


@pytest.fixture
def entities():
    cards_df = pd.DataFrame({"id": [1, 2, 3, 4, 5]})
    transactions_df = pd.DataFrame({"id": [1, 2, 3, 4, 5, 6],
                                    "card_id": [1, 2, 1, 3, 4, 5],
                                    "transaction_time": [10, 12, 13, 20, 21, 20],
                                    "fraud": [True, False, False, False, True, True]})
    entities = {
        "cards": (cards_df, "id"),
        "transactions": (transactions_df, "id", "transaction_time")
    }
    return entities


@pytest.fixture
def relationships():
    return [("cards", "id", "transactions", "card_id")]


def test_accepts_cutoff_time_df(entities, relationships):
    cutoff_times_df = pd.DataFrame({"instance_id": [1, 2, 3],
                                    "time": [10, 12, 15]})
    feature_matrix, features = dfs(entities=entities,
                                   relationships=relationships,
                                   target_entity="transactions",
                                   cutoff_time=cutoff_times_df)
    assert len(feature_matrix.index) == 3
    assert len(feature_matrix.columns) == len(features)


def test_accepts_single_cutoff_time(entities, relationships):
    feature_matrix, features = dfs(entities=entities,
                                   relationships=relationships,
                                   target_entity="transactions",
                                   cutoff_time=20)
    assert len(feature_matrix.index) == 5
    assert len(feature_matrix.columns) == len(features)


def test_accepts_no_cutoff_time(entities, relationships):
    feature_matrix, features = dfs(entities=entities,
                                   relationships=relationships,
                                   target_entity="transactions",
                                   instance_ids=[1, 2, 3, 5, 6])
    assert len(feature_matrix.index) == 5
    assert len(feature_matrix.columns) == len(features)


def test_ignores_instance_ids_if_cutoff_df(entities, relationships):
    cutoff_times_df = pd.DataFrame({"instance_id": [1, 2, 3],
                                    "time": [10, 12, 15]})
    instance_ids = [1, 2, 3, 4, 5]
    feature_matrix, features = dfs(entities=entities,
                                   relationships=relationships,
                                   target_entity="transactions",
                                   cutoff_time=cutoff_times_df,
                                   instance_ids=instance_ids)
    assert len(feature_matrix.index) == 3
    assert len(feature_matrix.columns) == len(features)


def test_approximate_features(entities, relationships):
    cutoff_times_df = pd.DataFrame({"instance_id": [1, 3, 1, 5, 3, 6],
                                    "time": [11, 16, 16, 26, 17, 22]})
    feature_matrix, features = dfs(entities=entities,
                                   relationships=relationships,
                                   target_entity="transactions",
                                   cutoff_time=cutoff_times_df,
                                   approximate=5,
                                   cutoff_time_in_index=True)
    direct_agg_feat_name = 'cards.PERCENT_TRUE(transactions.fraud)'
    assert len(feature_matrix.index) == 6
    assert len(feature_matrix.columns) == len(features)
    truth_index = pd.MultiIndex.from_arrays([[1, 3, 1, 5, 3, 6],
                                             [11, 16, 16, 26, 17, 22]],
                                            names=('id', 'time'))
    truth_values = pd.Series(data=[1.0, 0.5, 0.5, 1.0, 0.5, 1.0],
                             index=truth_index)
    truth_values.sort_index(level='time', kind='mergesort', inplace=True)

    assert (feature_matrix[direct_agg_feat_name] == truth_values).all()


def test_all_variables(entities, relationships):
    cutoff_times_df = pd.DataFrame({"instance_id": [1, 2, 3],
                                    "time": [10, 12, 15]})
    instance_ids = [1, 2, 3, 4, 5]
    feature_matrix, features = dfs(entities=entities,
                                   relationships=relationships,
                                   target_entity="transactions",
                                   cutoff_time=cutoff_times_df,
                                   instance_ids=instance_ids,
                                   agg_primitives=[Max, Mean, Min, Sum],
                                   trans_primitives=[],
                                   groupby_trans_primitives=["cum_sum"],
                                   max_depth=3,
                                   allowed_paths=None,
                                   ignore_entities=None,
                                   ignore_variables=None,
                                   seed_features=None)
    assert len(feature_matrix.index) == 3
    assert len(feature_matrix.columns) == len(features)


def test_features_only(entities, relationships):
    features = dfs(entities=entities,
                   relationships=relationships,
                   target_entity="transactions",
                   features_only=True)
    assert len(features) > 0


def test_dask_kwargs(entities, relationships):
    cutoff_times_df = pd.DataFrame({"instance_id": [1, 2, 3],
                                    "time": [10, 12, 15]})
    feature_matrix, features = dfs(entities=entities,
                                   relationships=relationships,
                                   target_entity="transactions",
                                   cutoff_time=cutoff_times_df)

    with cluster() as (scheduler, [a, b]):
        dask_kwargs = {'cluster': scheduler['address']}
        feature_matrix_2, features_2 = dfs(entities=entities,
                                           relationships=relationships,
                                           target_entity="transactions",
                                           cutoff_time=cutoff_times_df,
                                           dask_kwargs=dask_kwargs)

    assert all(f1.unique_name() == f2.unique_name() for f1, f2 in zip(features, features_2))
    for column in feature_matrix:
        for x, y in zip(feature_matrix[column], feature_matrix_2[column]):
            assert ((pd.isnull(x) and pd.isnull(y)) or (x == y))


def test_accepts_pandas_training_window():
    cards_df = pd.DataFrame({"id": [1, 2, 3, 4, 5]})
    transactions_df = pd.DataFrame({"id": [1, 2, 3, 4, 5],
                                    "card_id": [1, 1, 5, 1, 5],
                                    "transaction_time": pd.to_datetime([
                                        '2011-1-1 04:00', '2012-2-28 05:00',
                                        '2012-2-29 06:00', '2012-3-1 08:00',
                                        '2014-4-1 10:00']),
                                    "fraud": [True, False, False, False, True]})

    es = EntitySet(id="fraud_data")
    es = es.entity_from_dataframe(entity_id="transactions",
                                  dataframe=transactions_df,
                                  index="id",
                                  time_index="transaction_time")

    es = es.entity_from_dataframe(entity_id="cards",
                                  dataframe=cards_df,
                                  index="id")
    relationship = Relationship(es["cards"]["id"], es["transactions"]["card_id"])
    es = es.add_relationship(relationship)
    es.add_last_time_indexes()

    feature_matrix, features = dfs(entityset=es,
                                   target_entity="transactions")

    feature_matrix_2, features_2 = dfs(entityset=es,
                                       target_entity="transactions",
                                       cutoff_time=pd.Timestamp("2012-4-1 04:00"))

    feature_matrix_3, features_3 = dfs(entityset=es,
                                       target_entity="transactions",
                                       cutoff_time=pd.Timestamp("2012-4-1 04:00"),
                                       training_window=pd.Timedelta(3, "M"))

    # Test case for leap years
    feature_matrix_4, features_4 = dfs(entityset=es,
                                       target_entity="transactions",
                                       cutoff_time=pd.Timestamp("2013-2-28 04:00"),
                                       training_window=pd.Timedelta(1, "Y"))

    assert feature_matrix.index.all([1, 2, 3, 4, 5])
    assert feature_matrix_2.index.all([1, 2, 3, 4])
    assert feature_matrix_3.index.all([2, 3, 4])
    assert feature_matrix_4.index.all([2, 3])
