import numpy as np
import pandas as pd
import pytest
from distributed.utils_test import cluster

from featuretools.computational_backends.calculate_feature_matrix import (
    FEATURE_CALCULATION_PERCENTAGE
)
from featuretools.entityset import EntitySet, Relationship, Timedelta
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


@pytest.fixture
def datetime_es():
    cards_df = pd.DataFrame({"id": [1, 2, 3, 4, 5]})
    transactions_df = pd.DataFrame({"id": [1, 2, 3, 4, 5],
                                    "card_id": [1, 1, 5, 1, 5],
                                    "transaction_time": pd.to_datetime([
                                        '2011-2-28 04:00', '2012-2-28 05:00',
                                        '2012-2-29 06:00', '2012-3-1 08:00',
                                        '2014-4-1 10:00']),
                                    "fraud": [True, False, False, False, True]})

    datetime_es = EntitySet(id="fraud_data")
    datetime_es = datetime_es.entity_from_dataframe(entity_id="transactions",
                                                    dataframe=transactions_df,
                                                    index="id",
                                                    time_index="transaction_time")

    datetime_es = datetime_es.entity_from_dataframe(entity_id="cards",
                                                    dataframe=cards_df,
                                                    index="id")
    relationship = Relationship(datetime_es["cards"]["id"], datetime_es["transactions"]["card_id"])
    datetime_es = datetime_es.add_relationship(relationship)
    datetime_es.add_last_time_indexes()
    return datetime_es


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

    truth_values = pd.Series(data=[1.0, 0.5, 0.5, 1.0, 0.5, 1.0])

    assert (feature_matrix[direct_agg_feat_name] == truth_values.values).all()


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


def test_accepts_relative_training_window(datetime_es):
    feature_matrix, features = dfs(entityset=datetime_es,
                                   target_entity="transactions")

    feature_matrix_2, features_2 = dfs(entityset=datetime_es,
                                       target_entity="transactions",
                                       cutoff_time=pd.Timestamp("2012-4-1 04:00"))

    feature_matrix_3, features_3 = dfs(entityset=datetime_es,
                                       target_entity="transactions",
                                       cutoff_time=pd.Timestamp("2012-4-1 04:00"),
                                       training_window=Timedelta("3 months"))

    feature_matrix_4, features_4 = dfs(entityset=datetime_es,
                                       target_entity="transactions",
                                       cutoff_time=pd.Timestamp("2012-4-1 04:00"),
                                       training_window="3 months")

    # Test case for leap years
    feature_matrix_5, features_5 = dfs(entityset=datetime_es,
                                       target_entity="transactions",
                                       cutoff_time=pd.Timestamp("2012-2-29 04:00"),
                                       training_window=Timedelta("1 year"))

    assert (feature_matrix.index == [1, 2, 3, 4, 5]).all()
    assert (feature_matrix_2.index == [1, 2, 3, 4]).all()
    assert (feature_matrix_3.index == [2, 3, 4]).all()
    assert (feature_matrix_4.index == [2, 3, 4]).all()
    assert (feature_matrix_5.index == [1, 2]).all()


def test_accepts_pd_timedelta_training_window(datetime_es):
    feature_matrix, features = dfs(entityset=datetime_es,
                                   target_entity="transactions",
                                   cutoff_time=pd.Timestamp("2012-3-31 04:00"),
                                   training_window=pd.Timedelta(61, "D"))

    assert (feature_matrix.index == [2, 3, 4]).all()


def test_accepts_pd_dateoffset_training_window(datetime_es):
    feature_matrix, features = dfs(entityset=datetime_es,
                                   target_entity="transactions",
                                   cutoff_time=pd.Timestamp("2012-3-31 04:00"),
                                   training_window=pd.DateOffset(months=2))

    feature_matrix_2, features_2 = dfs(entityset=datetime_es,
                                       target_entity="transactions",
                                       cutoff_time=pd.Timestamp("2012-3-31 04:00"),
                                       training_window=pd.offsets.BDay(44))

    assert (feature_matrix.index == [2, 3, 4]).all()
    assert (feature_matrix.index == feature_matrix_2.index).all()


def test_calls_progress_callback(entities, relationships):
    class MockProgressCallback:
        def __init__(self):
            self.progress_history = []
            self.total_update = 0
            self.total_progress_percent = 0

        def __call__(self, update, progress_percent, time_elapsed):
            self.total_update += update
            self.total_progress_percent = progress_percent
            self.progress_history.append(progress_percent)

    mock_progress_callback = MockProgressCallback()

    feature_matrix, features = dfs(entities=entities,
                                   relationships=relationships,
                                   target_entity="transactions",
                                   progress_callback=mock_progress_callback)

    # second to last entry is the last update from feature calculation
    assert np.isclose(mock_progress_callback.progress_history[-2], FEATURE_CALCULATION_PERCENTAGE * 100)
    assert np.isclose(mock_progress_callback.total_update, 100.0)
    assert np.isclose(mock_progress_callback.total_progress_percent, 100.0)

    # test with multiple jobs
    mock_progress_callback = MockProgressCallback()

    with cluster() as (scheduler, [a, b]):
        dkwargs = {'cluster': scheduler['address']}
        feature_matrix, features = dfs(entities=entities,
                                       relationships=relationships,
                                       target_entity="transactions",
                                       progress_callback=mock_progress_callback,
                                       dask_kwargs=dkwargs)

    assert np.isclose(mock_progress_callback.total_update, 100.0)
    assert np.isclose(mock_progress_callback.total_progress_percent, 100.0)
