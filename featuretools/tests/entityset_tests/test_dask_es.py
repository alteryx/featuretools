import dask.dataframe as dd
import pandas as pd
import pytest

import featuretools as ft
from featuretools.entityset import EntitySet, Relationship


def test_create_entity_from_dask_df(pd_es):
    dask_es = EntitySet(id="dask_es")
    log_dask = dd.from_pandas(pd_es["log"].df, npartitions=2)
    dask_es = dask_es.entity_from_dataframe(
        entity_id="log_dask",
        dataframe=log_dask,
        index="id",
        time_index="datetime",
        variable_types=pd_es["log"].variable_types
    )
    pd.testing.assert_frame_equal(pd_es["log"].df, dask_es["log_dask"].df.compute(), check_like=True)


def test_create_entity_with_non_numeric_index(pd_es, dask_es):
    df = pd.DataFrame({"id": ["A_1", "A_2", "C", "D"],
                       "values": [1, 12, -34, 27]})
    dask_df = dd.from_pandas(df, npartitions=2)

    pd_es.entity_from_dataframe(
        entity_id="new_entity",
        dataframe=df,
        index="id")

    dask_es.entity_from_dataframe(
        entity_id="new_entity",
        dataframe=dask_df,
        index="id",
        variable_types={"id": ft.variable_types.Id, "values": ft.variable_types.Numeric})

    pd.testing.assert_frame_equal(pd_es['new_entity'].df.reset_index(drop=True), dask_es['new_entity'].df.compute())


def test_create_entityset_with_mixed_dataframe_types(pd_es, dask_es):
    df = pd.DataFrame({"id": [0, 1, 2, 3],
                       "values": [1, 12, -34, 27]})
    dask_df = dd.from_pandas(df, npartitions=2)

    # Test error is raised when trying to add Dask entity to entitset with existing pandas entities
    err_msg = "All entity dataframes must be of the same type. " \
              "Cannot add entity of type {} to an entityset with existing entities " \
              "of type {}".format(type(dask_df), type(pd_es.entities[0].df))

    with pytest.raises(ValueError, match=err_msg):
        pd_es.entity_from_dataframe(
            entity_id="new_entity",
            dataframe=dask_df,
            index="id")

    # Test error is raised when trying to add pandas entity to entitset with existing dask entities
    err_msg = "All entity dataframes must be of the same type. " \
              "Cannot add entity of type {} to an entityset with existing entities " \
              "of type {}".format(type(df), type(dask_es.entities[0].df))

    with pytest.raises(ValueError, match=err_msg):
        dask_es.entity_from_dataframe(
            entity_id="new_entity",
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
    sessions_vtypes = {
        "id": ft.variable_types.Id,
        "user": ft.variable_types.Id,
        "time": ft.variable_types.DatetimeTimeIndex,
        "strings": ft.variable_types.NaturalLanguage
    }

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
    transactions_vtypes = {
        "id": ft.variable_types.Id,
        "session_id": ft.variable_types.Id,
        "amount": ft.variable_types.Numeric,
        "time": ft.variable_types.DatetimeTimeIndex,
    }

    pd_es.entity_from_dataframe(entity_id="sessions", dataframe=sessions, index="id", time_index="time")
    dask_es.entity_from_dataframe(entity_id="sessions", dataframe=sessions_dask, index="id", time_index="time", variable_types=sessions_vtypes)

    pd_es.entity_from_dataframe(entity_id="transactions", dataframe=transactions, index="id", time_index="time")
    dask_es.entity_from_dataframe(entity_id="transactions", dataframe=transactions_dask, index="id", time_index="time", variable_types=transactions_vtypes)

    new_rel = Relationship(pd_es["sessions"]["id"],
                           pd_es["transactions"]["session_id"])
    dask_rel = Relationship(dask_es["sessions"]["id"],
                            dask_es["transactions"]["session_id"])

    pd_es = pd_es.add_relationship(new_rel)
    dask_es = dask_es.add_relationship(dask_rel)

    assert pd_es['sessions'].last_time_index is None
    assert dask_es['sessions'].last_time_index is None

    pd_es.add_last_time_indexes()
    dask_es.add_last_time_indexes()

    pd.testing.assert_series_equal(pd_es['sessions'].last_time_index.sort_index(), dask_es['sessions'].last_time_index.compute(), check_names=False)


def test_create_entity_with_make_index():
    values = [1, 12, -23, 27]
    df = pd.DataFrame({"values": values})
    dask_df = dd.from_pandas(df, npartitions=2)
    dask_es = EntitySet(id="dask_es")
    vtypes = {"values": ft.variable_types.Numeric}
    dask_es.entity_from_dataframe(entity_id="new_entity", dataframe=dask_df, make_index=True, index="new_index", variable_types=vtypes)

    expected_df = pd.DataFrame({"new_index": range(len(values)), "values": values})
    pd.testing.assert_frame_equal(expected_df, dask_es['new_entity'].df.compute())
