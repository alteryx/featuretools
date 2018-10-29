import copy
from datetime import datetime

import pandas as pd
import pytest

from ..testing_utils import make_ecommerce_entityset

from featuretools import Relationship


@pytest.fixture
def entityset():
    return make_ecommerce_entityset()


@pytest.fixture
def values_es(entityset):
    new_es = copy.deepcopy(entityset)
    new_es.normalize_entity('log', 'values', 'value',
                            make_time_index=True,
                            new_entity_time_index="value_time")
    return new_es


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
def extra_session_df(entityset):
    row_values = {'customer_id': 2,
                  'device_name': 'PC',
                  'device_type': 0,
                  'id': 6}
    row = pd.DataFrame(row_values, index=pd.Index([6], name='id'))
    df = entityset['sessions'].df.append(row, sort=True).sort_index()
    return df


class TestLastTimeIndex(object):
    def test_leaf(self, entityset):
        entityset.add_last_time_indexes()
        log = entityset['log']
        assert len(log.last_time_index) == 17
        for v1, v2 in zip(log.last_time_index, log.df['datetime']):
            assert (pd.isnull(v1) and pd.isnull(v2)) or v1 == v2

    def test_leaf_no_time_index(self, entityset):
        entityset.add_last_time_indexes()
        stores = entityset['stores']
        true_lti = pd.Series([None for x in range(6)], dtype='datetime64[ns]')
        assert len(true_lti) == len(stores.last_time_index)
        for v1, v2 in zip(stores.last_time_index, true_lti):
            assert (pd.isnull(v1) and pd.isnull(v2)) or v1 == v2

    def test_parent(self, values_es, true_values_lti):
        # test entity with time index and all instances in child entity
        values_es.add_last_time_indexes()
        values = values_es['values']
        assert len(values.last_time_index) == 11
        sorted_lti = values.last_time_index.sort_index()
        for v1, v2 in zip(sorted_lti, true_values_lti):
            assert (pd.isnull(v1) and pd.isnull(v2)) or v1 == v2

    def test_parent_some_missing(self, values_es, true_values_lti):
        # test entity with time index and not all instances have children
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

    def test_parent_no_time_index(self, entityset, true_sessions_lti):
        # test entity without time index and all instances have children
        entityset.add_last_time_indexes()
        sessions = entityset['sessions']
        assert len(sessions.last_time_index) == 6
        sorted_lti = sessions.last_time_index.sort_index()
        for v1, v2 in zip(sorted_lti, true_sessions_lti):
            assert (pd.isnull(v1) and pd.isnull(v2)) or v1 == v2

    def test_parent_no_time_index_missing(self, entityset, extra_session_df,
                                          true_sessions_lti):
        # test entity without time index and not all instance have children
        sessions = entityset['sessions']

        # add session instance with no associated log instances
        sessions.update_data(extra_session_df)
        entityset.add_last_time_indexes()
        # since sessions has no time index, default value is NaT
        true_sessions_lti[6] = pd.NaT

        assert len(sessions.last_time_index) == 7
        sorted_lti = sessions.last_time_index.sort_index()
        for v1, v2 in zip(sorted_lti, true_sessions_lti):
            assert (pd.isnull(v1) and pd.isnull(v2)) or v1 == v2

    def test_multiple_children(self, entityset, wishlist_df,
                               true_sessions_lti):
        # test all instances in both children
        entityset.entity_from_dataframe(entity_id="wishlist_log",
                                        dataframe=wishlist_df,
                                        index='id',
                                        make_index=True,
                                        time_index='datetime')
        relationship = Relationship(entityset['sessions']['id'],
                                    entityset['wishlist_log']['session_id'])
        entityset.add_relationship(relationship)
        entityset.add_last_time_indexes()
        sessions = entityset['sessions']
        # wishlist df has more recent events for two session ids
        true_sessions_lti[1] = pd.Timestamp("2011-4-9 10:31:30")
        true_sessions_lti[3] = pd.Timestamp("2011-4-10 10:41:00")

        assert len(sessions.last_time_index) == 6
        sorted_lti = sessions.last_time_index.sort_index()
        for v1, v2 in zip(sorted_lti, true_sessions_lti):
            assert (pd.isnull(v1) and pd.isnull(v2)) or v1 == v2

    def test_multiple_children_right_missing(self, entityset, wishlist_df,
                                             true_sessions_lti):
        # test all instances in left child
        sessions = entityset['sessions']

        # drop wishlist instance related to id 3 so it's only in log
        wishlist_df.drop(4, inplace=True)
        entityset.entity_from_dataframe(entity_id="wishlist_log",
                                        dataframe=wishlist_df,
                                        index='id',
                                        make_index=True,
                                        time_index='datetime')
        relationship = Relationship(entityset['sessions']['id'],
                                    entityset['wishlist_log']['session_id'])
        entityset.add_relationship(relationship)
        entityset.add_last_time_indexes()

        # now only session id 1 has newer event in wishlist_log
        true_sessions_lti[1] = pd.Timestamp("2011-4-9 10:31:30")

        assert len(sessions.last_time_index) == 6
        sorted_lti = sessions.last_time_index.sort_index()
        for v1, v2 in zip(sorted_lti, true_sessions_lti):
            assert (pd.isnull(v1) and pd.isnull(v2)) or v1 == v2

    def test_multiple_children_left_missing(self, entityset, extra_session_df,
                                            wishlist_df, true_sessions_lti):
        # test all instances in right child
        sessions = entityset['sessions']

        # add row to sessions so not all session instances are in log
        sessions.update_data(extra_session_df)

        # add row to wishlist df so new session instance in in wishlist_log
        row_values = {'session_id': 6,
                      'datetime': pd.Timestamp("2011-04-11 11:11:11"),
                      'product_id': 'toothpaste'}
        row = pd.DataFrame(row_values, index=pd.RangeIndex(start=7, stop=8))
        df = wishlist_df.append(row)
        entityset.entity_from_dataframe(entity_id="wishlist_log",
                                        dataframe=df,
                                        index='id',
                                        make_index=True,
                                        time_index='datetime')
        relationship = Relationship(entityset['sessions']['id'],
                                    entityset['wishlist_log']['session_id'])
        entityset.add_relationship(relationship)
        entityset.add_last_time_indexes()

        # now wishlist_log has newer events for 3 session ids
        true_sessions_lti[1] = pd.Timestamp("2011-4-9 10:31:30")
        true_sessions_lti[3] = pd.Timestamp("2011-4-10 10:41:00")
        true_sessions_lti[6] = pd.Timestamp("2011-04-11 11:11:11")

        assert len(sessions.last_time_index) == 7
        sorted_lti = sessions.last_time_index.sort_index()
        for v1, v2 in zip(sorted_lti, true_sessions_lti):
            assert (pd.isnull(v1) and pd.isnull(v2)) or v1 == v2

    def test_multiple_children_all_combined(self, entityset, extra_session_df,
                                            wishlist_df, true_sessions_lti):
        # test some instances in right, some in left, all when combined
        sessions = entityset['sessions']

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
        entityset.entity_from_dataframe(entity_id="wishlist_log",
                                        dataframe=df,
                                        index='id',
                                        make_index=True,
                                        time_index='datetime')
        relationship = Relationship(entityset['sessions']['id'],
                                    entityset['wishlist_log']['session_id'])
        entityset.add_relationship(relationship)
        entityset.add_last_time_indexes()

        # wishlist has newer events for 2 sessions
        true_sessions_lti[1] = pd.Timestamp("2011-4-9 10:31:30")
        true_sessions_lti[6] = pd.Timestamp("2011-04-11 11:11:11")

        assert len(sessions.last_time_index) == 7
        sorted_lti = sessions.last_time_index.sort_index()
        for v1, v2 in zip(sorted_lti, true_sessions_lti):
            assert (pd.isnull(v1) and pd.isnull(v2)) or v1 == v2

    def test_multiple_children_both_missing(self, entityset, extra_session_df,
                                            wishlist_df, true_sessions_lti):
        # test all instances in neither child
        sessions = entityset['sessions']

        # add row to sessions to create session with no events
        sessions.update_data(extra_session_df)

        entityset.entity_from_dataframe(entity_id="wishlist_log",
                                        dataframe=wishlist_df,
                                        index='id',
                                        make_index=True,
                                        time_index='datetime')
        relationship = Relationship(entityset['sessions']['id'],
                                    entityset['wishlist_log']['session_id'])
        entityset.add_relationship(relationship)
        entityset.add_last_time_indexes()
        sessions = entityset['sessions']

        # wishlist has 2 newer events and one is NaT
        true_sessions_lti[1] = pd.Timestamp("2011-4-9 10:31:30")
        true_sessions_lti[3] = pd.Timestamp("2011-4-10 10:41:00")
        true_sessions_lti[6] = pd.NaT

        assert len(sessions.last_time_index) == 7
        sorted_lti = sessions.last_time_index.sort_index()
        for v1, v2 in zip(sorted_lti, true_sessions_lti):
            assert (pd.isnull(v1) and pd.isnull(v2)) or v1 == v2

    def test_grandparent(self, entityset):
        # test sorting by time works correctly across several generations
        log = entityset["log"]
        customers = entityset["customers"]

        # For one user, change a log event to be newer than the user's normal
        # last time index. This event should be from a different session than
        # the current last time index.
        log.df['datetime'][5] = pd.Timestamp("2011-4-09 10:40:01")
        log.df = (log.df.set_index('datetime', append=True)
                     .sort_index(level=[1, 0], kind="mergesort")
                     .reset_index('datetime', drop=False))
        log.update_data(log.df)
        entityset.add_last_time_indexes()

        true_customers_lti = pd.Series([datetime(2011, 4, 9, 10, 40, 1),
                                        datetime(2011, 4, 10, 10, 41, 6),
                                        datetime(2011, 4, 10, 11, 10, 3)])

        assert len(customers.last_time_index) == 3
        sorted_lti = customers.last_time_index.sort_index()
        for v1, v2 in zip(sorted_lti, true_customers_lti):
            assert (pd.isnull(v1) and pd.isnull(v2)) or v1 == v2
