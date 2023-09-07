from datetime import datetime

import pandas as pd
import pytest
from woodwork.logical_types import Categorical, Datetime, Integer

from featuretools.entityset.entityset import LTI_COLUMN_NAME
from featuretools.tests.testing_utils import to_pandas
from featuretools.utils.gen_utils import Library, import_or_none

dd = import_or_none("dask.dataframe")
ps = import_or_none("pyspark.pandas")


@pytest.fixture
def values_es(es):
    es.normalize_dataframe(
        "log",
        "values",
        "value",
        make_time_index=True,
        new_dataframe_time_index="value_time",
    )
    return es


@pytest.fixture
def true_values_lti():
    true_values_lti = pd.Series(
        [
            datetime(2011, 4, 10, 10, 41, 0),
            datetime(2011, 4, 9, 10, 31, 9),
            datetime(2011, 4, 9, 10, 31, 18),
            datetime(2011, 4, 9, 10, 31, 27),
            datetime(2011, 4, 10, 10, 40, 1),
            datetime(2011, 4, 10, 10, 41, 3),
            datetime(2011, 4, 9, 10, 30, 12),
            datetime(2011, 4, 10, 10, 41, 6),
            datetime(2011, 4, 9, 10, 30, 18),
            datetime(2011, 4, 9, 10, 30, 24),
            datetime(2011, 4, 10, 11, 10, 3),
        ],
    )
    return true_values_lti


@pytest.fixture
def true_sessions_lti():
    sessions_lti = pd.Series(
        [
            datetime(2011, 4, 9, 10, 30, 24),
            datetime(2011, 4, 9, 10, 31, 27),
            datetime(2011, 4, 9, 10, 40, 0),
            datetime(2011, 4, 10, 10, 40, 1),
            datetime(2011, 4, 10, 10, 41, 6),
            datetime(2011, 4, 10, 11, 10, 3),
        ],
    )
    return sessions_lti


@pytest.fixture
def wishlist_df():
    wishlist_df = pd.DataFrame(
        {
            "session_id": [0, 1, 2, 2, 3, 4, 5],
            "datetime": [
                datetime(2011, 4, 9, 10, 30, 15),
                datetime(2011, 4, 9, 10, 31, 30),
                datetime(2011, 4, 9, 10, 30, 30),
                datetime(2011, 4, 9, 10, 35, 30),
                datetime(2011, 4, 10, 10, 41, 0),
                datetime(2011, 4, 10, 10, 39, 59),
                datetime(2011, 4, 10, 11, 10, 2),
            ],
            "product_id": [
                "coke zero",
                "taco clock",
                "coke zero",
                "car",
                "toothpaste",
                "brown bag",
                "coke zero",
            ],
        },
    )
    return wishlist_df


@pytest.fixture
def extra_session_df(es):
    row_values = {"customer_id": 2, "device_name": "PC", "device_type": 0, "id": 6}
    row = pd.DataFrame(row_values, index=pd.Index([6], name="id"))
    df = to_pandas(es["sessions"])
    df = pd.concat([df, row]).sort_index()
    if es.dataframe_type == Library.DASK:
        df = dd.from_pandas(df, npartitions=3)
    elif es.dataframe_type == Library.SPARK:
        # Spark can't handle object dtypes
        df = df.astype("string")
        df = ps.from_pandas(df)
    return df


class TestLastTimeIndex(object):
    def test_leaf(self, es):
        es.add_last_time_indexes()
        log = es["log"]
        lti_name = log.ww.metadata.get("last_time_index")

        assert lti_name == LTI_COLUMN_NAME
        assert len(log[lti_name]) == 17

        log_df = to_pandas(log)

        for v1, v2 in zip(log_df[lti_name], log_df["datetime"]):
            assert (pd.isnull(v1) and pd.isnull(v2)) or v1 == v2

    def test_leaf_no_time_index(self, es):
        es.add_last_time_indexes()
        stores = es["stores"]
        true_lti = pd.Series([None for x in range(6)], dtype="datetime64[ns]")

        assert len(true_lti) == len(stores[LTI_COLUMN_NAME])

        stores_lti = to_pandas(stores[LTI_COLUMN_NAME])

        for v1, v2 in zip(stores_lti, true_lti):
            assert (pd.isnull(v1) and pd.isnull(v2)) or v1 == v2

    # TODO: possible issue with either normalize_dataframe or add_last_time_indexes
    def test_parent(self, values_es, true_values_lti):
        # test dataframe with time index and all instances in child dataframe
        if values_es.dataframe_type != Library.PANDAS:
            pytest.xfail(
                "possible issue with either normalize_dataframe or add_last_time_indexes",
            )
        values_es.add_last_time_indexes()
        values = values_es["values"]
        lti_name = values.ww.metadata.get("last_time_index")
        assert len(values[lti_name]) == 10
        sorted_lti = to_pandas(values[lti_name]).sort_index()
        for v1, v2 in zip(sorted_lti, true_values_lti):
            assert (pd.isnull(v1) and pd.isnull(v2)) or v1 == v2

    # TODO: fails with Dask, tests needs to be reworked
    def test_parent_some_missing(self, values_es, true_values_lti):
        # test dataframe with time index and not all instances have children
        if values_es.dataframe_type != Library.PANDAS:
            pytest.xfail("fails with Dask, tests needs to be reworked")
        values = values_es["values"]

        # add extra value instance with no children
        row_values = {
            "value": [21.0],
            "value_time": [pd.Timestamp("2011-04-10 11:10:02")],
        }
        # make sure index doesn't have same name as column to suppress pandas warning
        row = pd.DataFrame(row_values, index=pd.Index([21]))
        df = pd.concat([values, row])
        df = df.sort_values(by="value")
        df.index.name = None

        values_es.replace_dataframe(dataframe_name="values", df=df)
        values_es.add_last_time_indexes()
        # lti value should default to instance's time index
        true_values_lti[10] = pd.Timestamp("2011-04-10 11:10:02")
        true_values_lti[11] = pd.Timestamp("2011-04-10 11:10:03")

        values = values_es["values"]
        lti_name = values.ww.metadata.get("last_time_index")
        assert len(values[lti_name]) == 11
        sorted_lti = values[lti_name].sort_index()
        for v1, v2 in zip(sorted_lti, true_values_lti):
            assert (pd.isnull(v1) and pd.isnull(v2)) or v1 == v2

    def test_parent_no_time_index(self, es, true_sessions_lti):
        # test dataframe without time index and all instances have children
        es.add_last_time_indexes()
        sessions = es["sessions"]
        lti_name = sessions.ww.metadata.get("last_time_index")
        assert len(sessions[lti_name]) == 6
        sorted_lti = to_pandas(sessions[lti_name]).sort_index()
        for v1, v2 in zip(sorted_lti, true_sessions_lti):
            assert (pd.isnull(v1) and pd.isnull(v2)) or v1 == v2

    def test_parent_no_time_index_missing(
        self,
        es,
        extra_session_df,
        true_sessions_lti,
    ):
        # test dataframe without time index and not all instance have children

        # add session instance with no associated log instances
        es.replace_dataframe(dataframe_name="sessions", df=extra_session_df)
        es.add_last_time_indexes()
        # since sessions has no time index, default value is NaT
        true_sessions_lti[6] = pd.NaT
        sessions = es["sessions"]

        lti_name = sessions.ww.metadata.get("last_time_index")
        assert len(sessions[lti_name]) == 7
        sorted_lti = to_pandas(sessions[lti_name]).sort_index()
        for v1, v2 in zip(sorted_lti, true_sessions_lti):
            assert (pd.isnull(v1) and pd.isnull(v2)) or v1 == v2

    def test_multiple_children(self, es, wishlist_df, true_sessions_lti):
        if es.dataframe_type == Library.SPARK:
            pytest.xfail("Cannot make index on a Spark DataFrame")
        # test all instances in both children
        if es.dataframe_type == Library.DASK:
            wishlist_df = dd.from_pandas(wishlist_df, npartitions=2)
        logical_types = {
            "session_id": Integer,
            "datetime": Datetime,
            "product_id": Categorical,
        }
        es.add_dataframe(
            dataframe_name="wishlist_log",
            dataframe=wishlist_df,
            index="id",
            make_index=True,
            time_index="datetime",
            logical_types=logical_types,
        )
        es.add_relationship("sessions", "id", "wishlist_log", "session_id")
        es.add_last_time_indexes()
        sessions = es["sessions"]
        # wishlist df has more recent events for two session ids
        true_sessions_lti[1] = pd.Timestamp("2011-4-9 10:31:30")
        true_sessions_lti[3] = pd.Timestamp("2011-4-10 10:41:00")

        lti_name = sessions.ww.metadata.get("last_time_index")
        assert len(sessions[lti_name]) == 6
        sorted_lti = to_pandas(sessions[lti_name]).sort_index()
        for v1, v2 in zip(sorted_lti, true_sessions_lti):
            assert (pd.isnull(v1) and pd.isnull(v2)) or v1 == v2

    def test_multiple_children_right_missing(self, es, wishlist_df, true_sessions_lti):
        if es.dataframe_type == Library.SPARK:
            pytest.xfail("Cannot make index on a Spark DataFrame")
        # test all instances in left child

        # drop wishlist instance related to id 3 so it's only in log
        wishlist_df.drop(4, inplace=True)
        if es.dataframe_type == Library.DASK:
            wishlist_df = dd.from_pandas(wishlist_df, npartitions=2)
        logical_types = {
            "session_id": Integer,
            "datetime": Datetime,
            "product_id": Categorical,
        }
        es.add_dataframe(
            dataframe_name="wishlist_log",
            dataframe=wishlist_df,
            index="id",
            make_index=True,
            time_index="datetime",
            logical_types=logical_types,
        )
        es.add_relationship("sessions", "id", "wishlist_log", "session_id")
        es.add_last_time_indexes()
        sessions = es["sessions"]

        # now only session id 1 has newer event in wishlist_log
        true_sessions_lti[1] = pd.Timestamp("2011-4-9 10:31:30")

        lti_name = sessions.ww.metadata.get("last_time_index")
        assert len(sessions[lti_name]) == 6
        sorted_lti = to_pandas(sessions[lti_name]).sort_index()
        for v1, v2 in zip(sorted_lti, true_sessions_lti):
            assert (pd.isnull(v1) and pd.isnull(v2)) or v1 == v2

    def test_multiple_children_left_missing(
        self,
        es,
        extra_session_df,
        wishlist_df,
        true_sessions_lti,
    ):
        if es.dataframe_type == Library.SPARK:
            pytest.xfail("Cannot make index on a Spark DataFrame")

        # add row to sessions so not all session instances are in log
        es.replace_dataframe(dataframe_name="sessions", df=extra_session_df)

        # add row to wishlist df so new session instance in in wishlist_log
        row_values = {
            "session_id": [6],
            "datetime": [pd.Timestamp("2011-04-11 11:11:11")],
            "product_id": ["toothpaste"],
        }
        row = pd.DataFrame(row_values, index=pd.RangeIndex(start=7, stop=8))
        df = pd.concat([wishlist_df, row])
        if es.dataframe_type == Library.DASK:
            df = dd.from_pandas(df, npartitions=2)
        logical_types = {
            "session_id": Integer,
            "datetime": Datetime,
            "product_id": Categorical,
        }
        es.add_dataframe(
            dataframe_name="wishlist_log",
            dataframe=df,
            index="id",
            make_index=True,
            time_index="datetime",
            logical_types=logical_types,
        )
        es.add_relationship("sessions", "id", "wishlist_log", "session_id")
        es.add_last_time_indexes()

        # test all instances in right child
        sessions = es["sessions"]

        # now wishlist_log has newer events for 3 session ids
        true_sessions_lti[1] = pd.Timestamp("2011-4-9 10:31:30")
        true_sessions_lti[3] = pd.Timestamp("2011-4-10 10:41:00")
        true_sessions_lti[6] = pd.Timestamp("2011-04-11 11:11:11")

        lti_name = sessions.ww.metadata.get("last_time_index")
        assert len(sessions[lti_name]) == 7
        sorted_lti = to_pandas(sessions[lti_name]).sort_index()
        for v1, v2 in zip(sorted_lti, true_sessions_lti):
            assert (pd.isnull(v1) and pd.isnull(v2)) or v1 == v2

    def test_multiple_children_all_combined(
        self,
        es,
        extra_session_df,
        wishlist_df,
        true_sessions_lti,
    ):
        if es.dataframe_type == Library.SPARK:
            pytest.xfail("Cannot make index on a Spark DataFrame")

        # add row to sessions so not all session instances are in log
        es.replace_dataframe(dataframe_name="sessions", df=extra_session_df)

        # add row to wishlist_log so extra session has child instance
        row_values = {
            "session_id": [6],
            "datetime": [pd.Timestamp("2011-04-11 11:11:11")],
            "product_id": ["toothpaste"],
        }
        row = pd.DataFrame(row_values, index=pd.RangeIndex(start=7, stop=8))
        df = pd.concat([wishlist_df, row])

        # drop instance 4 so wishlist_log does not have session id 3 instance
        df.drop(4, inplace=True)
        if es.dataframe_type == Library.DASK:
            df = dd.from_pandas(df, npartitions=2)
        logical_types = {
            "session_id": Integer,
            "datetime": Datetime,
            "product_id": Categorical,
        }
        es.add_dataframe(
            dataframe_name="wishlist_log",
            dataframe=df,
            index="id",
            make_index=True,
            time_index="datetime",
            logical_types=logical_types,
        )
        es.add_relationship("sessions", "id", "wishlist_log", "session_id")
        es.add_last_time_indexes()

        # test some instances in right, some in left, all when combined
        sessions = es["sessions"]

        # wishlist has newer events for 2 sessions
        true_sessions_lti[1] = pd.Timestamp("2011-4-9 10:31:30")
        true_sessions_lti[6] = pd.Timestamp("2011-04-11 11:11:11")

        lti_name = sessions.ww.metadata.get("last_time_index")
        assert len(sessions[lti_name]) == 7
        sorted_lti = to_pandas(sessions[lti_name]).sort_index()
        for v1, v2 in zip(sorted_lti, true_sessions_lti):
            assert (pd.isnull(v1) and pd.isnull(v2)) or v1 == v2

    def test_multiple_children_both_missing(
        self,
        es,
        extra_session_df,
        wishlist_df,
        true_sessions_lti,
    ):
        if es.dataframe_type == Library.SPARK:
            pytest.xfail("Cannot make index on a Spark DataFrame")
        # test all instances in neither child
        sessions = es["sessions"]

        if es.dataframe_type == Library.DASK:
            wishlist_df = dd.from_pandas(wishlist_df, npartitions=2)

        logical_types = {
            "session_id": Integer,
            "datetime": Datetime,
            "product_id": Categorical,
        }
        # add row to sessions to create session with no events
        es.replace_dataframe(dataframe_name="sessions", df=extra_session_df)

        es.add_dataframe(
            dataframe_name="wishlist_log",
            dataframe=wishlist_df,
            index="id",
            make_index=True,
            time_index="datetime",
            logical_types=logical_types,
        )
        es.add_relationship("sessions", "id", "wishlist_log", "session_id")
        es.add_last_time_indexes()
        sessions = es["sessions"]

        # wishlist has 2 newer events and one is NaT
        true_sessions_lti[1] = pd.Timestamp("2011-4-9 10:31:30")
        true_sessions_lti[3] = pd.Timestamp("2011-4-10 10:41:00")
        true_sessions_lti[6] = pd.NaT

        lti_name = sessions.ww.metadata.get("last_time_index")
        assert len(sessions[lti_name]) == 7
        sorted_lti = to_pandas(sessions[lti_name]).sort_index()
        for v1, v2 in zip(sorted_lti, true_sessions_lti):
            assert (pd.isnull(v1) and pd.isnull(v2)) or v1 == v2

    def test_grandparent(self, es):
        # test sorting by time works correctly across several generations
        log = es["log"]

        # For one user, change a log event to be newer than the user's normal
        # last time index. This event should be from a different session than
        # the current last time index.
        df = to_pandas(log)
        df["datetime"][5] = pd.Timestamp("2011-4-09 10:40:01")
        df = (
            df.set_index("datetime", append=True)
            .sort_index(level=[1, 0], kind="mergesort")
            .reset_index("datetime", drop=False)
        )
        if es.dataframe_type == Library.DASK:
            df = dd.from_pandas(df, npartitions=2)
        if es.dataframe_type == Library.SPARK:
            df = ps.from_pandas(df)
        es.replace_dataframe(dataframe_name="log", df=df)
        es.add_last_time_indexes()
        customers = es["customers"]

        true_customers_lti = pd.Series(
            [
                datetime(2011, 4, 9, 10, 40, 1),
                datetime(2011, 4, 10, 10, 41, 6),
                datetime(2011, 4, 10, 11, 10, 3),
            ],
        )

        lti_name = customers.ww.metadata.get("last_time_index")
        assert len(customers[lti_name]) == 3
        sorted_lti = to_pandas(customers).sort_values("id")[lti_name]
        for v1, v2 in zip(sorted_lti, true_customers_lti):
            assert (pd.isnull(v1) and pd.isnull(v2)) or v1 == v2
