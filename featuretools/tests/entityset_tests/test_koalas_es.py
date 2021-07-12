import pandas as pd
import pytest
from woodwork.logical_types import Datetime, Double, Integer, NaturalLanguage

from featuretools.entityset import EntitySet
from featuretools.tests.testing_utils import get_df_tags
from featuretools.utils.gen_utils import Library, import_or_none
from featuretools.utils.koalas_utils import pd_to_ks_clean

ks = import_or_none('databricks.koalas')


@pytest.mark.skipif('not ks')
def test_create_entity_from_ks_df(pd_es):
    cleaned_df = pd_to_ks_clean(pd_es["log"])
    log_ks = ks.from_pandas(cleaned_df)

    ks_es = EntitySet(id="ks_es")
    ks_es = ks_es.add_dataframe(
        dataframe_name="log_ks",
        dataframe=log_ks,
        index="id",
        time_index="datetime",
        logical_types=pd_es["log"].ww.logical_types,
        semantic_tags=get_df_tags(pd_es["log"])
    )
    pd.testing.assert_frame_equal(cleaned_df, ks_es["log_ks"].to_pandas(), check_like=True)


@pytest.mark.skipif('not ks')
def test_add_dataframe_with_non_numeric_index(pd_es, ks_es):
    df = pd.DataFrame({"id": pd.Series(["A_1", "A_2", "C", "D"], dtype='string'),
                       "values": [1, 12, -34, 27]})
    ks_df = ks.from_pandas(df)

    pd_es.add_dataframe(
        dataframe_name="new_entity",
        dataframe=df,
        index="id",
        logical_types={"id": NaturalLanguage, "values": Integer})

    ks_es.add_dataframe(
        dataframe_name="new_entity",
        dataframe=ks_df,
        index="id",
        logical_types={"id": NaturalLanguage, "values": Integer})
    pd.testing.assert_frame_equal(pd_es['new_entity'].reset_index(drop=True), ks_es['new_entity'].to_pandas())


@pytest.mark.skipif('not ks')
def test_create_entityset_with_mixed_dataframe_types(pd_es, ks_es):
    df = pd.DataFrame({"id": [0, 1, 2, 3],
                       "values": [1, 12, -34, 27]})
    ks_df = ks.from_pandas(df)

    err_msg = "All dataframes must be of the same type. " \
              "Cannot add dataframe of type {} to an entityset with existing dataframes " \
              "of type {}"

    # Test error is raised when trying to add Koalas entity to entitset with existing pandas entities
    with pytest.raises(ValueError, match=err_msg.format(type(ks_df), type(pd_es.dataframes[0]))):
        pd_es.add_dataframe(
            dataframe_name="new_entity",
            dataframe=ks_df,
            index="id")

    # Test error is raised when trying to add pandas entity to entitset with existing ks entities
    with pytest.raises(ValueError, match=err_msg.format(type(df), type(ks_es.dataframes[0]))):
        ks_es.add_dataframe(
            dataframe_name="new_entity",
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
    sessions_logical_types = {
        "id": Integer,
        "user": Integer,
        "strings": NaturalLanguage,
        "time": Datetime,
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
    transactions_logical_types = {
        "id": Integer,
        "session_id": Integer,
        "amount": Double,
        "time": Datetime,
    }

    pd_es.add_dataframe(dataframe_name="sessions", dataframe=sessions, index="id", time_index="time")
    ks_es.add_dataframe(dataframe_name="sessions", dataframe=sessions_ks, index="id", time_index="time",
                        logical_types=sessions_logical_types)

    pd_es.add_dataframe(dataframe_name="transactions", dataframe=transactions, index="id", time_index="time")
    ks_es.add_dataframe(dataframe_name="transactions", dataframe=transactions_ks, index="id", time_index="time",
                        logical_types=transactions_logical_types)

    pd_es = pd_es.add_relationship("sessions", "id", "transactions", "session_id")
    ks_es = ks_es.add_relationship("sessions", "id", "transactions", "session_id")

    assert 'foreign_key' in pd_es['transactions'].ww.semantic_tags['session_id']
    assert 'foreign_key' in ks_es['transactions'].ww.semantic_tags['session_id']

    assert pd_es['sessions'].ww.metadata.get('last_time_index') is None
    assert ks_es['sessions'].ww.metadata.get('last_time_index') is None

    pd_es.add_last_time_indexes()
    ks_es.add_last_time_indexes()

    pd_lti_name = pd_es['sessions'].ww.metadata.get('last_time_index')
    ks_lti_name = ks_es['sessions'].ww.metadata.get('last_time_index')
    assert pd_lti_name == ks_lti_name
    pd.testing.assert_series_equal(pd_es['sessions'][pd_lti_name].sort_index(),
                                   ks_es['sessions'][ks_lti_name].to_pandas().sort_index(), check_names=False)


@pytest.mark.skipif('not ks')
def test_add_dataframe_with_make_index():
    values = [1, 12, -23, 27]
    df = pd.DataFrame({"values": values})
    ks_df = ks.from_pandas(df)
    ks_es = EntitySet(id="ks_es")
    ltypes = {"values": "Integer"}
    ks_es.add_dataframe(dataframe_name="new_entity", dataframe=ks_df, make_index=True, index="new_index", logical_types=ltypes)

    expected_df = pd.DataFrame({"values": values, "new_index": range(len(values))})
    pd.testing.assert_frame_equal(expected_df, ks_es['new_entity'].to_pandas())


@pytest.mark.skipif('not ks')
def test_dataframe_type_koalas(ks_es):
    assert ks_es.dataframe_type == Library.KOALAS.value
