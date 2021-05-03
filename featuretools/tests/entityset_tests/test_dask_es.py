import dask.dataframe as dd
import pandas as pd
import pytest

import woodwork.logical_types as ltypes

import featuretools as ft
from featuretools.entityset import EntitySet
from featuretools.tests.testing_utils import get_df_tags


def test_add_dataframe(pd_es):
    dask_es = EntitySet(id="dask_es")
    log_dask = dd.from_pandas(pd_es["log"], npartitions=2)
    dask_es = dask_es.add_dataframe(
        dataframe_id="log_dask",
        dataframe=log_dask,
        index="id",
        time_index="datetime",
        logical_types=pd_es["log"].ww.logical_types,
        semantic_tags=get_df_tags(pd_es["log"])
    )
    pd.testing.assert_frame_equal(pd_es["log"], dask_es["log_dask"].compute(), check_like=True)


def test_add_dataframe_with_non_numeric_index(pd_es, dask_es):
    df = pd.DataFrame({"id": ["A_1", "A_2", "C", "D"],
                       "values": [1, 12, -34, 27]})
    dask_df = dd.from_pandas(df, npartitions=2)

    pd_es.add_dataframe(
        dataframe_id="new_entity",
        dataframe=df,
        index="id")

    dask_es.add_dataframe(
        dataframe_id="new_entity",
        dataframe=dask_df,
        index="id",
        logical_types={"id": ltypes.Categorical, "values": ltypes.Integer},
        semantic_tags={'id': 'foreign_key'})

    pd.testing.assert_frame_equal(pd_es['new_entity'].reset_index(drop=True), dask_es['new_entity'].compute())


def test_create_entityset_with_mixed_dataframe_types(pd_es, dask_es):
    df = pd.DataFrame({"id": [0, 1, 2, 3],
                       "values": [1, 12, -34, 27]})
    dask_df = dd.from_pandas(df, npartitions=2)

    # Test error is raised when trying to add Dask dataframe to entitset with existing pandas entities
    err_msg = "All dataframes must be of the same type. " \
              "Cannot add dataframe of type {} to an entityset with existing dataframes " \
              "of type {}".format(type(dask_df), type(pd_es.dataframes[0]))

    with pytest.raises(ValueError, match=err_msg):
        pd_es.add_dataframe(
            dataframe_id="new_dataframe",
            dataframe=dask_df,
            index="id")

    # Test error is raised when trying to add pandas dataframe to entitset with existing dask entities
    err_msg = "All dataframes must be of the same type. " \
              "Cannot add dataframe of type {} to an entityset with existing dataframes " \
              "of type {}".format(type(df), type(dask_es.dataframes[0]))

    with pytest.raises(ValueError, match=err_msg):
        dask_es.add_dataframe(
            dataframe_id="new_dataframe",
            dataframe=df,
            index="id")


def test_add_last_time_indexes():
    pd_es = EntitySet(id="pd_es")
    dask_es = EntitySet(id="dask_es")

    sessions = pd.DataFrame({"id": [0, 1, 2, 3],
                             "user": [1, 2, 1, 3],
                             "time": [pd.to_datetime('2019-01-10'),
                                      pd.to_datetime('2019-02-03'),
                                      pd.to_datetime('2019-01-01'),
                                      pd.to_datetime('2017-08-25')],
                             "strings": ["I am a string",
                                         "23",
                                         "abcdef ghijk",
                                         ""]})
    sessions_dask = dd.from_pandas(sessions, npartitions=2)
    sessions_logical_types = {
        "id": ltypes.Integer,
        "user": ltypes.Integer,
        "time": ltypes.Datetime,
        "strings": ltypes.NaturalLanguage
    }
    sessions_semantic_tags = {'user': 'foreign_key'}

    transactions = pd.DataFrame({"id": [0, 1, 2, 3, 4, 5],
                                 "session_id": [0, 0, 1, 2, 2, 3],
                                 "amount": [1.23, 5.24, 123.52, 67.93, 40.34, 50.13],
                                 "time": [pd.to_datetime('2019-01-10 03:53'),
                                          pd.to_datetime('2019-01-10 04:12'),
                                          pd.to_datetime('2019-02-03 10:34'),
                                          pd.to_datetime('2019-01-01 12:35'),
                                          pd.to_datetime('2019-01-01 12:49'),
                                          pd.to_datetime('2017-08-25 04:53')]})
    transactions_dask = dd.from_pandas(transactions, npartitions=2)

    transactions_logical_types = {
        "id": ltypes.Integer,
        "session_id": ltypes.Integer,
        "time": ltypes.Datetime,
        "amount": ltypes.Double
    }
    transactions_semantic_tags = {'session_id': 'foreign_key'}

    pd_es.add_dataframe(dataframe_id="sessions", dataframe=sessions, index="id", time_index="time")
    dask_es.add_dataframe(dataframe_id="sessions", dataframe=sessions_dask,
                          index="id", time_index="time",
                          logical_types=sessions_logical_types, semantic_tags=sessions_semantic_tags)

    pd_es.add_dataframe(dataframe_id="transactions", dataframe=transactions, index="id", time_index="time")
    dask_es.add_dataframe(dataframe_id="transactions", dataframe=transactions_dask,
                          index="id", time_index="time",
                          logical_types=transactions_logical_types, semantic_tags=transactions_semantic_tags)

    pd_es = pd_es.add_relationship("sessions", "id", "transactions", "session_id")
    dask_es = dask_es.add_relationship("sessions", "id", "transactions", "session_id")

    assert pd_es['sessions'].ww.metadata.get('last_time_index') is None
    assert dask_es['sessions'].ww.metadata.get('last_time_index') is None

    pd_es.add_last_time_indexes()
    dask_es.add_last_time_indexes()

    pd.testing.assert_series_equal(pd_es['sessions'].ww.metadata.get('last_time_index').sort_index(),
                                   dask_es['sessions'].ww.metadata.get('last_time_index').compute(), check_names=False)


def test_add_dataframe_with_make_index():
    values = [1, 12, -23, 27]
    df = pd.DataFrame({"values": values})
    dask_df = dd.from_pandas(df, npartitions=2)
    dask_es = EntitySet(id="dask_es")
    logical_types = {"values": ltypes.Integer}
    dask_es.add_dataframe(dataframe_id="new_entity", dataframe=dask_df, make_index=True, index="new_index", logical_types=logical_types)

    expected_df = pd.DataFrame({"values": values, "new_index": range(len(values))})
    pd.testing.assert_frame_equal(expected_df, dask_es['new_entity'].compute())
