from datetime import datetime

import pandas as pd
import pytest
from dask import dataframe as dd

import featuretools as ft
from featuretools import Relationship
from featuretools.tests.testing_utils import to_pandas
from featuretools.utils.gen_utils import import_or_none

ks = import_or_none('databricks.koalas')


@pytest.fixture
def values_es(es):
    es.normalize_entity('log', 'values', 'value',
                        make_time_index=True,
                        new_entity_time_index="value_time")
    return es


@pytest.fixture
def true_values_lti():
    true_values_lti = pd.Series([datetime(2011, 4, 10, 10, 41, 0),
                                 datetime(2011, 4, 9, 10, 31, 9),
                                 datetime(2011, 4, 9, 10, 31, 18),
                                 datetime(2011, 4, 9, 10, 31, 27),
                                 datetime(2011, 4, 10, 10, 40, 1),
                                 datetime(2011, 4, 10, 10, 41, 3),
                                 datetime(2011, 4, 9, 10, 30, 12),
                                 datetime(2011, 4, 10, 10, 41, 6),
                                 datetime(2011, 4, 9, 10, 30, 18),
                                 datetime(2011, 4, 9, 10, 30, 24),
                                 datetime(2011, 4, 10, 11, 10, 3)])
    return true_values_lti


@pytest.fixture
def true_sessions_lti():
    sessions_lti = pd.Series([datetime(2011, 4, 9, 10, 30, 24),
                              datetime(2011, 4, 9, 10, 31, 27),
                              datetime(2011, 4, 9, 10, 40, 0),
                              datetime(2011, 4, 10, 10, 40, 1),
                              datetime(2011, 4, 10, 10, 41, 6),
                              datetime(2011, 4, 10, 11, 10, 3)])
    return sessions_lti


@pytest.fixture
def wishlist_df():
    wishlist_df = pd.DataFrame({
        "session_id": [0, 1, 2, 2, 3, 4, 5],
        "datetime": [datetime(2011, 4, 9, 10, 30, 15),
                     datetime(2011, 4, 9, 10, 31, 30),
                     datetime(2011, 4, 9, 10, 30, 30),
                     datetime(2011, 4, 9, 10, 35, 30),
                     datetime(2011, 4, 10, 10, 41, 0),
                     datetime(2011, 4, 10, 10, 39, 59),
                     datetime(2011, 4, 10, 11, 10, 2)],
        "product_id": ['coke zero', 'taco clock', 'coke zero', 'car',
                       'toothpaste', 'brown bag', 'coke zero'],
    })
    return wishlist_df


@pytest.fixture
def extra_session_df(es):
    row_values = {'customer_id': 2,
                  'device_name': 'PC',
                  'device_type': 0,
                  'id': 6}
    row = pd.DataFrame(row_values, index=pd.Index([6], name='id'))
    df = to_pandas(es['sessions'].df)
    df = df.append(row, sort=True).sort_index()
    if isinstance(es['sessions'].df, dd.DataFrame):
        df = dd.from_pandas(df, npartitions=3)
    if ks and isinstance(es['sessions'].df, ks.DataFrame):
        df = ks.from_pandas(df)
    return df


class TestLastTimeIndex(object):
    def test_leaf(self, es):
        es.add_last_time_indexes()
        log = es['log']
        assert len(log.last_time_index) == 17
        log_df = to_pandas(log.df)
        log_lti = to_pandas(log.last_time_index)
        for v1, v2 in zip(log_lti, log_df['datetime']):
            assert (pd.isnull(v1) and pd.isnull(v2)) or v1 == v2

    def test_leaf_no_time_index(self, es):
        es.add_last_time_indexes()
        stores = es['stores']
        true_lti = pd.Series([None for x in range(6)], dtype='datetime64[ns]')
        assert len(true_lti) == len(stores.last_time_index)
        stores_lti = to_pandas(stores.last_time_index)
        for v1, v2 in zip(stores_lti, true_lti):
            assert (pd.isnull(v1) and pd.isnull(v2)) or v1 == v2

    # TODO: possible issue with either normalize_entity or add_last_time_indexes
    def test_parent(self, values_es, true_values_lti):
        # test entity with time index and all instances in child entity
        if not all(isinstance(entity.df, pd.DataFrame) for entity in values_es.entities):
            pytest.xfail('possible issue with either normalize_entity or add_last_time_indexes')
        values_es.add_last_time_indexes()
        values = values_es['values']
        assert len(values.last_time_index) == 11
        sorted_lti = to_pandas(values.last_time_index).sort_index()
        for v1, v2 in zip(sorted_lti, true_values_lti):
            assert (pd.isnull(v1) and pd.isnull(v2)) or v1 == v2

    # TODO: fails with Dask, tests needs to be reworked
    def test_parent_some_missing(self, values_es, true_values_lti):
        # test entity with time index and not all instances have children
        if not all(isinstance(entity.df, pd.DataFrame) for entity in values_es.entities):
            pytest.xfail('fails with Dask, tests needs to be reworked')
        values = values_es['values']

        # add extra value instance with no children
        row_values = {'value': 21.0,
                      'value_time': pd.Timestamp("2011-04-10 11:10:02"),
                      'values_id': 11}
        # make sure index doesn't have same name as column to suppress pandas warning
        row = pd.DataFrame(row_values, index=pd.Index([11]))
        df = values.df.append(row, sort=True)
        df = df[['value', 'value_time']].sort_values(by='value')
        df.index.name = 'values_id'
        values.update_data(df)
        values_es.add_last_time_indexes()
        # lti value should default to instance's time index
        true_values_lti[10] = pd.Timestamp("2011-04-10 11:10:02")
        true_values_lti[11] = pd.Timestamp("2011-04-10 11:10:03")

        assert len(values.last_time_index) == 12
        sorted_lti = values.last_time_index.sort_index()
        for v1, v2 in zip(sorted_lti, true_values_lti):
            assert (pd.isnull(v1) and pd.isnull(v2)) or v1 == v2

    def test_parent_no_time_index(self, es, true_sessions_lti):
        # test entity without time index and all instances have children
        es.add_last_time_indexes()
        sessions = es['sessions']
        assert len(sessions.last_time_index) == 6
        sorted_lti = to_pandas(sessions.last_time_index).sort_index()
        for v1, v2 in zip(sorted_lti, true_sessions_lti):
            assert (pd.isnull(v1) and pd.isnull(v2)) or v1 == v2

    def test_parent_no_time_index_missing(self, es, extra_session_df,
                                          true_sessions_lti):
        # test entity without time index and not all instance have children
        sessions = es['sessions']

        # add session instance with no associated log instances
        sessions.update_data(extra_session_df)
        es.add_last_time_indexes()
        # since sessions has no time index, default value is NaT
        true_sessions_lti[6] = pd.NaT

        assert len(sessions.last_time_index) == 7
        sorted_lti = to_pandas(sessions.last_time_index).sort_index()
        for v1, v2 in zip(sorted_lti, true_sessions_lti):
            assert (pd.isnull(v1) and pd.isnull(v2)) or v1 == v2

    def test_multiple_children(self, es, wishlist_df,
                               true_sessions_lti):
        # test all instances in both children
        if isinstance(es.entities[0].df, dd.DataFrame):
            wishlist_df = dd.from_pandas(wishlist_df, npartitions=2)
        if ks and isinstance(es.entities[0].df, ks.DataFrame):
            wishlist_df = ks.from_pandas(wishlist_df)
        variable_types = {'id': ft.variable_types.variable.Index,
                          'session_id': ft.variable_types.variable.Numeric,
                          'datetime': ft.variable_types.variable.DatetimeTimeIndex,
                          'product_id': ft.variable_types.variable.Categorical}
        es.entity_from_dataframe(entity_id="wishlist_log",
                                 dataframe=wishlist_df,
                                 index='id',
                                 make_index=True,
                                 time_index='datetime',
                                 variable_types=variable_types)
        relationship = Relationship(es['sessions']['id'],
                                    es['wishlist_log']['session_id'])
        es.add_relationship(relationship)
        es.add_last_time_indexes()
        sessions = es['sessions']
        # wishlist df has more recent events for two session ids
        true_sessions_lti[1] = pd.Timestamp("2011-4-9 10:31:30")
        true_sessions_lti[3] = pd.Timestamp("2011-4-10 10:41:00")

        assert len(sessions.last_time_index) == 6
        sorted_lti = to_pandas(sessions.last_time_index).sort_index()
        for v1, v2 in zip(sorted_lti, true_sessions_lti):
            assert (pd.isnull(v1) and pd.isnull(v2)) or v1 == v2

    def test_multiple_children_right_missing(self, es, wishlist_df,
                                             true_sessions_lti):
        # test all instances in left child
        sessions = es['sessions']

        # drop wishlist instance related to id 3 so it's only in log
        wishlist_df.drop(4, inplace=True)
        if isinstance(es.entities[0].df, dd.DataFrame):
            wishlist_df = dd.from_pandas(wishlist_df, npartitions=2)
        if ks and isinstance(es.entities[0].df, ks.DataFrame):
            wishlist_df = ks.from_pandas(wishlist_df)
        variable_types = {'id': ft.variable_types.variable.Index,
                          'session_id': ft.variable_types.variable.Numeric,
                          'datetime': ft.variable_types.variable.DatetimeTimeIndex,
                          'product_id': ft.variable_types.variable.Categorical}
        es.entity_from_dataframe(entity_id="wishlist_log",
                                 dataframe=wishlist_df,
                                 index='id',
                                 make_index=True,
                                 time_index='datetime',
                                 variable_types=variable_types)
        relationship = Relationship(es['sessions']['id'],
                                    es['wishlist_log']['session_id'])
        es.add_relationship(relationship)
        es.add_last_time_indexes()

        # now only session id 1 has newer event in wishlist_log
        true_sessions_lti[1] = pd.Timestamp("2011-4-9 10:31:30")

        assert len(sessions.last_time_index) == 6
        sorted_lti = to_pandas(sessions.last_time_index).sort_index()
        for v1, v2 in zip(sorted_lti, true_sessions_lti):
            assert (pd.isnull(v1) and pd.isnull(v2)) or v1 == v2

    def test_multiple_children_left_missing(self, es, extra_session_df,
                                            wishlist_df, true_sessions_lti):
        # test all instances in right child
        sessions = es['sessions']

        # add row to sessions so not all session instances are in log
        sessions.update_data(extra_session_df)

        # add row to wishlist df so new session instance in in wishlist_log
        row_values = {'session_id': 6,
                      'datetime': pd.Timestamp("2011-04-11 11:11:11"),
                      'product_id': 'toothpaste'}
        row = pd.DataFrame(row_values, index=pd.RangeIndex(start=7, stop=8))
        df = wishlist_df.append(row)
        if isinstance(es.entities[0].df, dd.DataFrame):
            df = dd.from_pandas(df, npartitions=2)
        if ks and isinstance(es.entities[0].df, ks.DataFrame):
            df = ks.from_pandas(df)
        variable_types = {'id': ft.variable_types.variable.Index,
                          'session_id': ft.variable_types.variable.Numeric,
                          'datetime': ft.variable_types.variable.DatetimeTimeIndex,
                          'product_id': ft.variable_types.variable.Categorical}
        es.entity_from_dataframe(entity_id="wishlist_log",
                                 dataframe=df,
                                 index='id',
                                 make_index=True,
                                 time_index='datetime',
                                 variable_types=variable_types)
        relationship = Relationship(es['sessions']['id'],
                                    es['wishlist_log']['session_id'])
        es.add_relationship(relationship)
        es.add_last_time_indexes()

        # now wishlist_log has newer events for 3 session ids
        true_sessions_lti[1] = pd.Timestamp("2011-4-9 10:31:30")
        true_sessions_lti[3] = pd.Timestamp("2011-4-10 10:41:00")
        true_sessions_lti[6] = pd.Timestamp("2011-04-11 11:11:11")

        assert len(sessions.last_time_index) == 7
        sorted_lti = to_pandas(sessions.last_time_index).sort_index()
        for v1, v2 in zip(sorted_lti, true_sessions_lti):
            assert (pd.isnull(v1) and pd.isnull(v2)) or v1 == v2

    def test_multiple_children_all_combined(self, es, extra_session_df,
                                            wishlist_df, true_sessions_lti):
        # test some instances in right, some in left, all when combined
        sessions = es['sessions']

        # add row to sessions so not all session instances are in log
        sessions.update_data(extra_session_df)

        # add row to wishlist_log so extra session has child instance
        row_values = {'session_id': 6,
                      'datetime': pd.Timestamp("2011-04-11 11:11:11"),
                      'product_id': 'toothpaste'}
        row = pd.DataFrame(row_values, index=pd.RangeIndex(start=7, stop=8))
        df = wishlist_df.append(row)

        # drop instance 4 so wishlist_log does not have session id 3 instance
        df.drop(4, inplace=True)
        if isinstance(es.entities[0].df, dd.DataFrame):
            df = dd.from_pandas(df, npartitions=2)
        if ks and isinstance(es.entities[0].df, ks.DataFrame):
            df = ks.from_pandas(df)
        variable_types = {'id': ft.variable_types.variable.Index,
                          'session_id': ft.variable_types.variable.Numeric,
                          'datetime': ft.variable_types.variable.DatetimeTimeIndex,
                          'product_id': ft.variable_types.variable.Categorical}
        es.entity_from_dataframe(entity_id="wishlist_log",
                                 dataframe=df,
                                 index='id',
                                 make_index=True,
                                 time_index='datetime',
                                 variable_types=variable_types)
        relationship = Relationship(es['sessions']['id'],
                                    es['wishlist_log']['session_id'])
        es.add_relationship(relationship)
        es.add_last_time_indexes()

        # wishlist has newer events for 2 sessions
        true_sessions_lti[1] = pd.Timestamp("2011-4-9 10:31:30")
        true_sessions_lti[6] = pd.Timestamp("2011-04-11 11:11:11")

        assert len(sessions.last_time_index) == 7
        sorted_lti = to_pandas(sessions.last_time_index).sort_index()
        for v1, v2 in zip(sorted_lti, true_sessions_lti):
            assert (pd.isnull(v1) and pd.isnull(v2)) or v1 == v2

    def test_multiple_children_both_missing(self, es, extra_session_df,
                                            wishlist_df, true_sessions_lti):
        # test all instances in neither child
        sessions = es['sessions']

        if isinstance(es.entities[0].df, dd.DataFrame):
            wishlist_df = dd.from_pandas(wishlist_df, npartitions=2)
        if ks and isinstance(es.entities[0].df, ks.DataFrame):
            wishlist_df = ks.from_pandas(wishlist_df)

        variable_types = {'id': ft.variable_types.variable.Index,
                          'session_id': ft.variable_types.variable.Numeric,
                          'datetime': ft.variable_types.variable.DatetimeTimeIndex,
                          'product_id': ft.variable_types.variable.Categorical}
        # add row to sessions to create session with no events
        sessions.update_data(extra_session_df)

        es.entity_from_dataframe(entity_id="wishlist_log",
                                 dataframe=wishlist_df,
                                 index='id',
                                 make_index=True,
                                 time_index='datetime',
                                 variable_types=variable_types)
        relationship = Relationship(es['sessions']['id'],
                                    es['wishlist_log']['session_id'])
        es.add_relationship(relationship)
        es.add_last_time_indexes()
        sessions = es['sessions']

        # wishlist has 2 newer events and one is NaT
        true_sessions_lti[1] = pd.Timestamp("2011-4-9 10:31:30")
        true_sessions_lti[3] = pd.Timestamp("2011-4-10 10:41:00")
        true_sessions_lti[6] = pd.NaT

        assert len(sessions.last_time_index) == 7
        sorted_lti = to_pandas(sessions.last_time_index).sort_index()
        for v1, v2 in zip(sorted_lti, true_sessions_lti):
            assert (pd.isnull(v1) and pd.isnull(v2)) or v1 == v2

    def test_grandparent(self, es):
        # test sorting by time works correctly across several generations
        log = es["log"]
        customers = es["customers"]

        # For one user, change a log event to be newer than the user's normal
        # last time index. This event should be from a different session than
        # the current last time index.
        df = to_pandas(log.df)
        df['datetime'][5] = pd.Timestamp("2011-4-09 10:40:01")
        df = (df.set_index('datetime', append=True)
              .sort_index(level=[1, 0], kind="mergesort")
              .reset_index('datetime', drop=False))
        if isinstance(log.df, dd.DataFrame):
            df = dd.from_pandas(df, npartitions=2)
        if ks and isinstance(log.df, ks.DataFrame):
            df = ks.from_pandas(df)
        log.update_data(df)
        es.add_last_time_indexes()

        true_customers_lti = pd.Series([datetime(2011, 4, 9, 10, 40, 1),
                                        datetime(2011, 4, 10, 10, 41, 6),
                                        datetime(2011, 4, 10, 11, 10, 3)])

        assert len(customers.last_time_index) == 3
        sorted_lti = to_pandas(customers.last_time_index).sort_index()
        for v1, v2 in zip(sorted_lti, true_customers_lti):
            assert (pd.isnull(v1) and pd.isnull(v2)) or v1 == v2
