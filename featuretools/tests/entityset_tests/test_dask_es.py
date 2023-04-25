import pandas as pd
import pytest
from woodwork.logical_types import (
    Categorical,
    Datetime,
    Double,
    Integer,
    NaturalLanguage,
)

from featuretools.entityset import EntitySet
from featuretools.tests.testing_utils import get_df_tags
from featuretools.utils.gen_utils import Library, import_or_none

dd = import_or_none("dask.dataframe")


@pytest.mark.skipif("not dd")
def test_add_dataframe(pd_es):
    dask_es = EntitySet(id="dask_es")
    log_dask = dd.from_pandas(pd_es["log"], npartitions=2)
    dask_es = dask_es.add_dataframe(
        dataframe_name="log_dask",
        dataframe=log_dask,
        index="id",
        time_index="datetime",
        logical_types=pd_es["log"].ww.logical_types,
        semantic_tags=get_df_tags(pd_es["log"]),
    )
    pd.testing.assert_frame_equal(
        pd_es["log"],
        dask_es["log_dask"].compute(),
        check_like=True,
    )


@pytest.mark.skipif("not dd")
def test_add_dataframe_with_non_numeric_index(pd_es, dask_es):
    df = pd.DataFrame({"id": ["A_1", "A_2", "C", "D"], "values": [1, 12, -34, 27]})
    dask_df = dd.from_pandas(df, npartitions=2)

    pd_es.add_dataframe(
        dataframe_name="new_dataframe",
        dataframe=df,
        index="id",
        logical_types={"id": Categorical, "values": Integer},
    )

    dask_es.add_dataframe(
        dataframe_name="new_dataframe",
        dataframe=dask_df,
        index="id",
        logical_types={"id": Categorical, "values": Integer},
    )

    pd.testing.assert_frame_equal(
        pd_es["new_dataframe"].reset_index(drop=True),
        dask_es["new_dataframe"].compute(),
    )


@pytest.mark.skipif("not dd")
def test_create_entityset_with_mixed_dataframe_types(pd_es, dask_es):
    df = pd.DataFrame({"id": [0, 1, 2, 3], "values": [1, 12, -34, 27]})
    dask_df = dd.from_pandas(df, npartitions=2)

    err_msg = (
        "All dataframes must be of the same type. "
        "Cannot add dataframe of type {} to an entityset with existing dataframes "
        "of type {}"
    )

    # Test error is raised when trying to add Dask dataframe to entityset with existing pandas dataframes
    with pytest.raises(
        ValueError,
        match=err_msg.format(type(dask_df), type(pd_es.dataframes[0])),
    ):
        pd_es.add_dataframe(
            dataframe_name="new_dataframe",
            dataframe=dask_df,
            index="id",
        )

    # Test error is raised when trying to add pandas dataframe to entityset with existing dask dataframes
    with pytest.raises(
        ValueError,
        match=err_msg.format(type(df), type(dask_es.dataframes[0])),
    ):
        dask_es.add_dataframe(dataframe_name="new_dataframe", dataframe=df, index="id")


@pytest.mark.skipif("not dd")
def test_add_last_time_indexes():
    pd_es = EntitySet(id="pd_es")
    dask_es = EntitySet(id="dask_es")

    sessions = pd.DataFrame(
        {
            "id": [0, 1, 2, 3],
            "user": [1, 2, 1, 3],
            "time": [
                pd.to_datetime("2019-01-10"),
                pd.to_datetime("2019-02-03"),
                pd.to_datetime("2019-01-01"),
                pd.to_datetime("2017-08-25"),
            ],
            "strings": ["I am a string", "23", "abcdef ghijk", ""],
        },
    )
    sessions_dask = dd.from_pandas(sessions, npartitions=2)
    sessions_logical_types = {
        "id": Integer,
        "user": Integer,
        "time": Datetime,
        "strings": NaturalLanguage,
    }

    transactions = pd.DataFrame(
        {
            "id": [0, 1, 2, 3, 4, 5],
            "session_id": [0, 0, 1, 2, 2, 3],
            "amount": [1.23, 5.24, 123.52, 67.93, 40.34, 50.13],
            "time": [
                pd.to_datetime("2019-01-10 03:53"),
                pd.to_datetime("2019-01-10 04:12"),
                pd.to_datetime("2019-02-03 10:34"),
                pd.to_datetime("2019-01-01 12:35"),
                pd.to_datetime("2019-01-01 12:49"),
                pd.to_datetime("2017-08-25 04:53"),
            ],
        },
    )
    transactions_dask = dd.from_pandas(transactions, npartitions=2)

    transactions_logical_types = {
        "id": Integer,
        "session_id": Integer,
        "time": Datetime,
        "amount": Double,
    }

    pd_es.add_dataframe(
        dataframe_name="sessions",
        dataframe=sessions,
        index="id",
        time_index="time",
    )
    dask_es.add_dataframe(
        dataframe_name="sessions",
        dataframe=sessions_dask,
        index="id",
        time_index="time",
        logical_types=sessions_logical_types,
    )

    pd_es.add_dataframe(
        dataframe_name="transactions",
        dataframe=transactions,
        index="id",
        time_index="time",
    )
    dask_es.add_dataframe(
        dataframe_name="transactions",
        dataframe=transactions_dask,
        index="id",
        time_index="time",
        logical_types=transactions_logical_types,
    )

    pd_es = pd_es.add_relationship("sessions", "id", "transactions", "session_id")
    dask_es = dask_es.add_relationship("sessions", "id", "transactions", "session_id")

    assert "foreign_key" in pd_es["transactions"].ww.semantic_tags["session_id"]
    assert "foreign_key" in dask_es["transactions"].ww.semantic_tags["session_id"]

    assert pd_es["sessions"].ww.metadata.get("last_time_index") is None
    assert dask_es["sessions"].ww.metadata.get("last_time_index") is None

    pd_es.add_last_time_indexes()
    dask_es.add_last_time_indexes()

    pd_lti_name = pd_es["sessions"].ww.metadata.get("last_time_index")
    spark_lti_name = dask_es["sessions"].ww.metadata.get("last_time_index")
    assert pd_lti_name == spark_lti_name
    pd.testing.assert_series_equal(
        pd_es["sessions"][pd_lti_name].sort_index(),
        dask_es["sessions"][spark_lti_name].compute().sort_index(),
        check_names=False,
    )


@pytest.mark.skipif("not dd")
def test_add_dataframe_with_make_index():
    values = [1, 12, -23, 27]
    df = pd.DataFrame({"values": values})
    dask_df = dd.from_pandas(df, npartitions=2)
    dask_es = EntitySet(id="dask_es")
    logical_types = {"values": Integer}
    dask_es.add_dataframe(
        dataframe_name="new_dataframe",
        dataframe=dask_df,
        make_index=True,
        index="new_index",
        logical_types=logical_types,
    )

    expected_df = pd.DataFrame({"values": values, "new_index": range(len(values))})
    pd.testing.assert_frame_equal(expected_df, dask_es["new_dataframe"].compute())


@pytest.mark.skipif("not dd")
def test_dataframe_type_dask(dask_es):
    assert dask_es.dataframe_type == Library.DASK
