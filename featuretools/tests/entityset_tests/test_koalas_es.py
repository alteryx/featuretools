import pandas as pd
import pytest

import featuretools as ft
from featuretools.entityset import EntitySet, Relationship
from featuretools.utils.gen_utils import import_or_none
from featuretools.utils.koalas_utils import pd_to_ks_clean

ks = import_or_none('databricks.koalas')


@pytest.mark.skipif('not ks')
def test_create_entity_from_ks_df(pd_es):
    cleaned_df = pd_to_ks_clean(pd_es["log"].df)
    log_ks = ks.from_pandas(cleaned_df)

    ks_es = EntitySet(id="ks_es")
    ks_es = ks_es.entity_from_dataframe(
        entity_id="log_ks",
        dataframe=log_ks,
        index="id",
        time_index="datetime",
        variable_types=pd_es["log"].variable_types
    )
    pd.testing.assert_frame_equal(cleaned_df, ks_es["log_ks"].df.to_pandas(), check_like=True)


@pytest.mark.skipif('not ks')
def test_create_entity_with_non_numeric_index(pd_es, ks_es):
    df = pd.DataFrame({"id": ["A_1", "A_2", "C", "D"],
                       "values": [1, 12, -34, 27]})
    ks_df = ks.from_pandas(df)

    pd_es.entity_from_dataframe(
        entity_id="new_entity",
        dataframe=df,
        index="id")

    ks_es.entity_from_dataframe(
        entity_id="new_entity",
        dataframe=ks_df,
        index="id",
        variable_types={"id": ft.variable_types.Id, "values": ft.variable_types.Numeric})
    pd.testing.assert_frame_equal(pd_es['new_entity'].df.reset_index(drop=True), ks_es['new_entity'].df.to_pandas())


@pytest.mark.skipif('not ks')
def test_create_entityset_with_mixed_dataframe_types(pd_es, ks_es):
    df = pd.DataFrame({"id": [0, 1, 2, 3],
                       "values": [1, 12, -34, 27]})
    ks_df = ks.from_pandas(df)

    # Test error is raised when trying to add Koalas entity to entitset with existing pandas entities
    err_msg = "All entity dataframes must be of the same type. " \
              "Cannot add entity of type {} to an entityset with existing entities " \
              "of type {}".format(type(ks_df), type(pd_es.entities[0].df))

    with pytest.raises(ValueError, match=err_msg):
        pd_es.entity_from_dataframe(
            entity_id="new_entity",
            dataframe=ks_df,
            index="id")

    # Test error is raised when trying to add pandas entity to entitset with existing ks entities
    err_msg = "All entity dataframes must be of the same type. " \
              "Cannot add entity of type {} to an entityset with existing entities " \
              "of type {}".format(type(df), type(ks_es.entities[0].df))

    with pytest.raises(ValueError, match=err_msg):
        ks_es.entity_from_dataframe(
            entity_id="new_entity",
            dataframe=df,
            index="id")


@pytest.mark.skipif('not ks')
def test_add_last_time_indexes():
    pd_es = EntitySet(id="pd_es")
    ks_es = EntitySet(id="ks_es")

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
    sessions_ks = ks.from_pandas(sessions)
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
    transactions_ks = ks.from_pandas(transactions)
    transactions_vtypes = {
        "id": ft.variable_types.Id,
        "session_id": ft.variable_types.Id,
        "amount": ft.variable_types.Numeric,
        "time": ft.variable_types.DatetimeTimeIndex,
    }

    pd_es.entity_from_dataframe(entity_id="sessions", dataframe=sessions, index="id", time_index="time")
    ks_es.entity_from_dataframe(entity_id="sessions", dataframe=sessions_ks, index="id", time_index="time", variable_types=sessions_vtypes)

    pd_es.entity_from_dataframe(entity_id="transactions", dataframe=transactions, index="id", time_index="time")
    ks_es.entity_from_dataframe(entity_id="transactions", dataframe=transactions_ks, index="id", time_index="time", variable_types=transactions_vtypes)

    new_rel = Relationship(pd_es["sessions"]["id"], pd_es["transactions"]["session_id"])
    ks_rel = Relationship(ks_es["sessions"]["id"], ks_es["transactions"]["session_id"])

    pd_es = pd_es.add_relationship(new_rel)
    ks_es = ks_es.add_relationship(ks_rel)

    assert pd_es['sessions'].last_time_index is None
    assert ks_es['sessions'].last_time_index is None

    pd_es.add_last_time_indexes()
    ks_es.add_last_time_indexes()

    pd.testing.assert_series_equal(pd_es['sessions'].last_time_index.sort_index(), ks_es['sessions'].last_time_index.to_pandas().sort_index(), check_names=False)


@pytest.mark.skipif('not ks')
def test_create_entity_with_make_index():
    values = [1, 12, -23, 27]
    df = pd.DataFrame({"values": values})
    ks_df = ks.from_pandas(df)
    ks_es = EntitySet(id="ks_es")
    vtypes = {"values": ft.variable_types.Numeric}
    ks_es.entity_from_dataframe(entity_id="new_entity", dataframe=ks_df, make_index=True, index="new_index", variable_types=vtypes)

    expected_df = pd.DataFrame({"new_index": range(len(values)), "values": values})
    pd.testing.assert_frame_equal(expected_df, ks_es['new_entity'].df.to_pandas().sort_index())
