import pandas as pd
import pytest
from woodwork.logical_types import Datetime, Double, Integer, NaturalLanguage

from featuretools.entityset import EntitySet
from featuretools.tests.testing_utils import get_df_tags
from featuretools.utils.gen_utils import Library, import_or_none
from featuretools.utils.spark_utils import pd_to_spark_clean

ps = import_or_none("pyspark.pandas")


@pytest.mark.skipif("not ps")
def test_add_dataframe_from_spark_df(pd_es):
    cleaned_df = pd_to_spark_clean(pd_es["log"])
    log_spark = ps.from_pandas(cleaned_df)

    spark_es = EntitySet(id="spark_es")
    spark_es = spark_es.add_dataframe(
        dataframe_name="log_spark",
        dataframe=log_spark,
        index="id",
        time_index="datetime",
        logical_types=pd_es["log"].ww.logical_types,
        semantic_tags=get_df_tags(pd_es["log"]),
    )
    pd.testing.assert_frame_equal(
        cleaned_df,
        spark_es["log_spark"].to_pandas(),
        check_like=True,
    )


@pytest.mark.skipif("not ps")
def test_add_dataframe_with_non_numeric_index(pd_es, spark_es):
    df = pd.DataFrame(
        {
            "id": pd.Series(["A_1", "A_2", "C", "D"], dtype="string"),
            "values": [1, 12, -34, 27],
        },
    )
    spark_df = ps.from_pandas(df)

    pd_es.add_dataframe(
        dataframe_name="new_dataframe",
        dataframe=df,
        index="id",
        logical_types={"id": NaturalLanguage, "values": Integer},
    )

    spark_es.add_dataframe(
        dataframe_name="new_dataframe",
        dataframe=spark_df,
        index="id",
        logical_types={"id": NaturalLanguage, "values": Integer},
    )
    pd.testing.assert_frame_equal(
        pd_es["new_dataframe"].reset_index(drop=True),
        spark_es["new_dataframe"].to_pandas(),
    )


@pytest.mark.skipif("not ps")
def test_create_entityset_with_mixed_dataframe_types(pd_es, spark_es):
    df = pd.DataFrame({"id": [0, 1, 2, 3], "values": [1, 12, -34, 27]})
    spark_df = ps.from_pandas(df)

    err_msg = (
        "All dataframes must be of the same type. "
        "Cannot add dataframe of type {} to an entityset with existing dataframes "
        "of type {}"
    )

    # Test error is raised when trying to add Spark dataframe to entitset with existing pandas dataframes
    with pytest.raises(
        ValueError,
        match=err_msg.format(type(spark_df), type(pd_es.dataframes[0])),
    ):
        pd_es.add_dataframe(
            dataframe_name="new_dataframe",
            dataframe=spark_df,
            index="id",
        )

    # Test error is raised when trying to add pandas dataframe to entitset with existing ps dataframes
    with pytest.raises(
        ValueError,
        match=err_msg.format(type(df), type(spark_es.dataframes[0])),
    ):
        spark_es.add_dataframe(dataframe_name="new_dataframe", dataframe=df, index="id")


@pytest.mark.skipif("not ps")
def test_add_last_time_indexes():
    pd_es = EntitySet(id="pd_es")
    spark_es = EntitySet(id="spark_es")

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
    sessions_spark = ps.from_pandas(sessions)
    sessions_logical_types = {
        "id": Integer,
        "user": Integer,
        "strings": NaturalLanguage,
        "time": Datetime,
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
    transactions_spark = ps.from_pandas(transactions)
    transactions_logical_types = {
        "id": Integer,
        "session_id": Integer,
        "amount": Double,
        "time": Datetime,
    }

    pd_es.add_dataframe(
        dataframe_name="sessions",
        dataframe=sessions,
        index="id",
        time_index="time",
    )
    spark_es.add_dataframe(
        dataframe_name="sessions",
        dataframe=sessions_spark,
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
    spark_es.add_dataframe(
        dataframe_name="transactions",
        dataframe=transactions_spark,
        index="id",
        time_index="time",
        logical_types=transactions_logical_types,
    )

    pd_es = pd_es.add_relationship("sessions", "id", "transactions", "session_id")
    spark_es = spark_es.add_relationship("sessions", "id", "transactions", "session_id")

    assert "foreign_key" in pd_es["transactions"].ww.semantic_tags["session_id"]
    assert "foreign_key" in spark_es["transactions"].ww.semantic_tags["session_id"]

    assert pd_es["sessions"].ww.metadata.get("last_time_index") is None
    assert spark_es["sessions"].ww.metadata.get("last_time_index") is None

    pd_es.add_last_time_indexes()
    spark_es.add_last_time_indexes()

    pd_lti_name = pd_es["sessions"].ww.metadata.get("last_time_index")
    spark_lti_name = spark_es["sessions"].ww.metadata.get("last_time_index")
    assert pd_lti_name == spark_lti_name
    pd.testing.assert_series_equal(
        pd_es["sessions"][pd_lti_name].sort_index(),
        spark_es["sessions"][spark_lti_name].to_pandas().sort_index(),
        check_names=False,
    )


@pytest.mark.skipif("not ps")
def test_add_dataframe_with_make_index():
    values = [1, 12, -23, 27]
    df = pd.DataFrame({"values": values})
    spark_df = ps.from_pandas(df)
    spark_es = EntitySet(id="spark_es")
    ltypes = {"values": "Integer"}
    spark_es.add_dataframe(
        dataframe_name="new_dataframe",
        dataframe=spark_df,
        make_index=True,
        index="new_index",
        logical_types=ltypes,
    )

    expected_df = pd.DataFrame({"values": values, "new_index": range(len(values))})
    pd.testing.assert_frame_equal(expected_df, spark_es["new_dataframe"].to_pandas())


@pytest.mark.skipif("not ps")
def test_dataframe_type_spark(spark_es):
    assert spark_es.dataframe_type == Library.SPARK
