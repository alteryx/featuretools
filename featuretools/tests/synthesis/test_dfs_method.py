import sys

import composeml as cp
import numpy as np
import pandas as pd
import pytest
from dask import dataframe as dd
from distributed.utils_test import cluster

from featuretools import variable_types as vtypes
from featuretools.computational_backends.calculate_feature_matrix import (
    FEATURE_CALCULATION_PERCENTAGE
)
from featuretools.entityset import EntitySet, Relationship, Timedelta
from featuretools.exceptions import UnusedPrimitiveWarning
from featuretools.primitives import (
    GreaterThanScalar,
    Max,
    Mean,
    Min,
    Sum,
    make_agg_primitive,
    make_trans_primitive
)
from featuretools.synthesis import dfs
from featuretools.tests.testing_utils import to_pandas
from featuretools.utils.gen_utils import import_or_none
from featuretools.variable_types import Numeric

ks = import_or_none('databricks.koalas')


@pytest.fixture(params=['pd_entities', 'dask_entities', 'koalas_entities'])
def entities(request):
    return request.getfixturevalue(request.param)


@pytest.fixture
def pd_entities():
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
def dask_entities():
    cards_df = pd.DataFrame({"id": [1, 2, 3, 4, 5]})
    transactions_df = pd.DataFrame({"id": [1, 2, 3, 4, 5, 6],
                                    "card_id": [1, 2, 1, 3, 4, 5],
                                    "transaction_time": [10, 12, 13, 20, 21, 20],
                                    "fraud": [True, False, False, False, True, True]})
    cards_df = dd.from_pandas(cards_df, npartitions=2)
    transactions_df = dd.from_pandas(transactions_df, npartitions=2)

    cards_vtypes = {
        'id': vtypes.Index
    }
    transactions_vtypes = {
        'id': vtypes.Index,
        'card_id': vtypes.Id,
        'transaction_time': vtypes.NumericTimeIndex,
        'fraud': vtypes.Boolean
    }

    entities = {
        "cards": (cards_df, "id", None, cards_vtypes),
        "transactions": (transactions_df, "id", "transaction_time", transactions_vtypes)
    }
    return entities


@pytest.fixture
def koalas_entities():
    ks = pytest.importorskip('databricks.koalas', reason="Koalas not installed, skipping")
    if sys.platform.startswith('win'):
        pytest.skip('skipping Koalas tests for Windows')
    cards_df = ks.DataFrame({"id": [1, 2, 3, 4, 5]})
    transactions_df = ks.DataFrame({"id": [1, 2, 3, 4, 5, 6],
                                    "card_id": [1, 2, 1, 3, 4, 5],
                                    "transaction_time": [10, 12, 13, 20, 21, 20],
                                    "fraud": [True, False, False, False, True, True]})
    cards_vtypes = {
        'id': vtypes.Index
    }
    transactions_vtypes = {
        'id': vtypes.Index,
        'card_id': vtypes.Id,
        'transaction_time': vtypes.NumericTimeIndex,
        'fraud': vtypes.Boolean
    }

    entities = {
        "cards": (cards_df, "id", None, cards_vtypes),
        "transactions": (transactions_df, "id", "transaction_time", transactions_vtypes)
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
    feature_matrix = to_pandas(feature_matrix, index='id', sort_index=True)
    assert len(feature_matrix.index) == 3
    assert len(feature_matrix.columns) == len(features)


def test_warns_cutoff_time_dask(entities, relationships):
    cutoff_times_df = pd.DataFrame({"instance_id": [1, 2, 3],
                                    "time": [10, 12, 15]})
    cutoff_times_df = dd.from_pandas(cutoff_times_df, npartitions=2)
    match = "cutoff_time should be a Pandas DataFrame: " \
            "computing cutoff_time, this may take a while"
    with pytest.warns(UserWarning, match=match):
        feature_matrix, features = dfs(entities=entities,
                                       relationships=relationships,
                                       target_entity="transactions",
                                       cutoff_time=cutoff_times_df)


def test_accepts_cutoff_time_compose(entities, relationships):
    def fraud_occured(df):
        return df['fraud'].any()

    lm = cp.LabelMaker(
        target_entity='card_id',
        time_index='transaction_time',
        labeling_function=fraud_occured,
        window_size=1
    )

    transactions_df = to_pandas(entities['transactions'][0])

    labels = lm.search(
        transactions_df,
        num_examples_per_instance=-1
    )

    labels['time'] = pd.to_numeric(labels['time'])
    labels.rename({'card_id': 'id'}, axis=1, inplace=True)

    feature_matrix, features = dfs(entities=entities,
                                   relationships=relationships,
                                   target_entity="cards",
                                   cutoff_time=labels)
    feature_matrix = to_pandas(feature_matrix, index='id')
    assert len(feature_matrix.index) == 6
    assert len(feature_matrix.columns) == len(features) + 1


def test_accepts_single_cutoff_time(entities, relationships):
    feature_matrix, features = dfs(entities=entities,
                                   relationships=relationships,
                                   target_entity="transactions",
                                   cutoff_time=20)
    feature_matrix = to_pandas(feature_matrix, index='id')
    assert len(feature_matrix.index) == 5
    assert len(feature_matrix.columns) == len(features)


def test_accepts_no_cutoff_time(entities, relationships):
    feature_matrix, features = dfs(entities=entities,
                                   relationships=relationships,
                                   target_entity="transactions",
                                   instance_ids=[1, 2, 3, 5, 6])
    feature_matrix = to_pandas(feature_matrix, index='id')
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
    feature_matrix = to_pandas(feature_matrix, index='id')
    assert len(feature_matrix.index) == 3
    assert len(feature_matrix.columns) == len(features)


def test_approximate_features(pd_entities, relationships):
    # TODO: Update to use Dask entities when issue #985 is closed
    cutoff_times_df = pd.DataFrame({"instance_id": [1, 3, 1, 5, 3, 6],
                                    "time": [11, 16, 16, 26, 17, 22]})
    feature_matrix, features = dfs(entities=pd_entities,
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


def test_all_variables(pd_entities, relationships):
    cutoff_times_df = pd.DataFrame({"instance_id": [1, 2, 3],
                                    "time": [10, 12, 15]})
    instance_ids = [1, 2, 3, 4, 5]
    feature_matrix, features = dfs(entities=pd_entities,
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


def test_accepts_relative_training_window(datetime_es):
    # TODO: Update to use Dask entities when issue #882 is closed
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

    assert (feature_matrix.index == [1, 2, 3, 4, 5]).all()
    assert (feature_matrix_2.index == [1, 2, 3, 4]).all()
    assert (feature_matrix_3.index == [2, 3, 4]).all()
    assert (feature_matrix_4.index == [2, 3, 4]).all()

    # Test case for leap years
    feature_matrix_5, features_5 = dfs(entityset=datetime_es,
                                       target_entity="transactions",
                                       cutoff_time=pd.Timestamp("2012-2-29 04:00"),
                                       training_window=Timedelta("1 year"),
                                       include_cutoff_time=True)
    assert (feature_matrix_5.index == [2]).all()

    feature_matrix_5, features_5 = dfs(entityset=datetime_es,
                                       target_entity="transactions",
                                       cutoff_time=pd.Timestamp("2012-2-29 04:00"),
                                       training_window=Timedelta("1 year"),
                                       include_cutoff_time=False)
    assert (feature_matrix_5.index == [1, 2]).all()


def test_accepts_pd_timedelta_training_window(datetime_es):
    # TODO: Update to use Dask entities when issue #882 is closed
    feature_matrix, features = dfs(entityset=datetime_es,
                                   target_entity="transactions",
                                   cutoff_time=pd.Timestamp("2012-3-31 04:00"),
                                   training_window=pd.Timedelta(61, "D"))

    assert (feature_matrix.index == [2, 3, 4]).all()


def test_accepts_pd_dateoffset_training_window(datetime_es):
    # TODO: Update to use Dask entities when issue #882 is closed
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


def test_warns_with_unused_primitives(es):
    if ks and any(isinstance(e.df, ks.DataFrame) for e in es.entities):
        pytest.skip('Koalas throws extra warnings')
    trans_primitives = ['num_characters', 'num_words', 'add_numeric']
    agg_primitives = [Max, 'min']

    warning_text = "Some specified primitives were not used during DFS:\n" + \
        "  trans_primitives: ['add_numeric']\n  agg_primitives: ['max', 'min']\n" + \
        "This may be caused by a using a value of max_depth that is too small, not setting interesting values, " + \
        "or it may indicate no compatible variable types for the primitive were found in the data."

    with pytest.warns(UnusedPrimitiveWarning) as record:
        dfs(entityset=es,
            target_entity='customers',
            trans_primitives=trans_primitives,
            agg_primitives=agg_primitives,
            max_depth=1)

    assert record[0].message.args[0] == warning_text

    # Should not raise a warning
    with pytest.warns(None) as record:
        dfs(entityset=es,
            target_entity='customers',
            trans_primitives=trans_primitives,
            agg_primitives=agg_primitives,
            max_depth=2)

    assert not record


def test_does_not_warn_with_stacking_feature(pd_es):
    with pytest.warns(None) as record:
        dfs(entityset=pd_es,
            target_entity='régions',
            agg_primitives=['percent_true'],
            trans_primitives=[GreaterThanScalar(5)],
            primitive_options={'greater_than_scalar': {'include_entities': ['stores']}},
            features_only=True)

    assert not record


def test_warns_with_unused_where_primitives(es):
    warning_text = "Some specified primitives were not used during DFS:\n" + \
        "  where_primitives: ['count', 'sum']\n" + \
        "This may be caused by a using a value of max_depth that is too small, not setting interesting values, " + \
        "or it may indicate no compatible variable types for the primitive were found in the data."

    with pytest.warns(UnusedPrimitiveWarning) as record:
        dfs(entityset=es,
            target_entity='customers',
            agg_primitives=['count'],
            where_primitives=['sum', 'count'],
            max_depth=1)

    assert record[0].message.args[0] == warning_text


def test_warns_with_unused_groupby_primitives(pd_es):
    warning_text = "Some specified primitives were not used during DFS:\n" + \
        "  groupby_trans_primitives: ['cum_sum']\n" + \
        "This may be caused by a using a value of max_depth that is too small, not setting interesting values, " + \
        "or it may indicate no compatible variable types for the primitive were found in the data."

    with pytest.warns(UnusedPrimitiveWarning) as record:
        dfs(entityset=pd_es,
            target_entity='sessions',
            groupby_trans_primitives=['cum_sum'],
            max_depth=1)

    assert record[0].message.args[0] == warning_text

    # Should not raise a warning
    with pytest.warns(None) as record:
        dfs(entityset=pd_es,
            target_entity='customers',
            groupby_trans_primitives=['cum_sum'],
            max_depth=1)

    assert not record


def test_warns_with_unused_custom_primitives(pd_es):
    def above_ten(column):
        return column > 10

    AboveTen = make_trans_primitive(function=above_ten,
                                    input_types=[Numeric],
                                    return_type=Numeric)

    trans_primitives = [AboveTen]

    warning_text = "Some specified primitives were not used during DFS:\n" + \
        "  trans_primitives: ['above_ten']\n" + \
        "This may be caused by a using a value of max_depth that is too small, not setting interesting values, " + \
        "or it may indicate no compatible variable types for the primitive were found in the data."

    with pytest.warns(UnusedPrimitiveWarning) as record:
        dfs(entityset=pd_es,
            target_entity='sessions',
            trans_primitives=trans_primitives,
            max_depth=1)

    assert record[0].message.args[0] == warning_text

    # Should not raise a warning
    with pytest.warns(None) as record:
        dfs(entityset=pd_es,
            target_entity='customers',
            trans_primitives=trans_primitives,
            max_depth=1)

    def max_above_ten(column):
        return max(column) > 10

    MaxAboveTen = make_agg_primitive(function=max_above_ten,
                                     input_types=[Numeric],
                                     return_type=Numeric)

    agg_primitives = [MaxAboveTen]

    warning_text = "Some specified primitives were not used during DFS:\n" + \
        "  agg_primitives: ['max_above_ten']\n" + \
        "This may be caused by a using a value of max_depth that is too small, not setting interesting values, " + \
        "or it may indicate no compatible variable types for the primitive were found in the data."

    with pytest.warns(UnusedPrimitiveWarning) as record:
        dfs(entityset=pd_es,
            target_entity='stores',
            agg_primitives=agg_primitives,
            max_depth=1)

    assert record[0].message.args[0] == warning_text

    # Should not raise a warning
    with pytest.warns(None) as record:
        dfs(entityset=pd_es,
            target_entity='sessions',
            agg_primitives=agg_primitives,
            max_depth=1)


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


def test_calls_progress_callback_cluster(pd_entities, relationships):
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

    with cluster() as (scheduler, [a, b]):
        dkwargs = {'cluster': scheduler['address']}
        feature_matrix, features = dfs(entities=pd_entities,
                                       relationships=relationships,
                                       target_entity="transactions",
                                       progress_callback=mock_progress_callback,
                                       dask_kwargs=dkwargs)

    assert np.isclose(mock_progress_callback.total_update, 100.0)
    assert np.isclose(mock_progress_callback.total_progress_percent, 100.0)


def test_dask_kwargs(pd_entities, relationships):
    cutoff_times_df = pd.DataFrame({"instance_id": [1, 2, 3],
                                    "time": [10, 12, 15]})
    feature_matrix, features = dfs(entities=pd_entities,
                                   relationships=relationships,
                                   target_entity="transactions",
                                   cutoff_time=cutoff_times_df)

    with cluster() as (scheduler, [a, b]):
        dask_kwargs = {'cluster': scheduler['address']}
        feature_matrix_2, features_2 = dfs(entities=pd_entities,
                                           relationships=relationships,
                                           target_entity="transactions",
                                           cutoff_time=cutoff_times_df,
                                           dask_kwargs=dask_kwargs)

    assert all(f1.unique_name() == f2.unique_name() for f1, f2 in zip(features, features_2))
    for column in feature_matrix:
        for x, y in zip(feature_matrix[column], feature_matrix_2[column]):
            assert ((pd.isnull(x) and pd.isnull(y)) or (x == y))
