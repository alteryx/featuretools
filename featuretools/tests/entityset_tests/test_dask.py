import dask.dataframe as dd
import pandas as pd
import pytest

import featuretools as ft
from featuretools.entityset import EntitySet, Relationship


def test_transform(es, dask_es):
    primitives = ft.list_primitives()
    trans_list = primitives[primitives['type'] == 'transform']['name'].tolist()
    # These primitives currently not supported with Dask
    not_supported = ['cum_mean', 'equal', 'not_equal', 'equal_scalar', 'not_equal_scalar']
    trans_primitives = [prim for prim in trans_list if prim not in not_supported]
    agg_primitives = []

    assert es == dask_es

    # Run DFS using each entity as a target and confirm results match
    for entity in es.entities:
        fm, _ = ft.dfs(entityset=es,
                       target_entity=entity.id,
                       trans_primitives=trans_primitives,
                       agg_primitives=agg_primitives,
                       cutoff_time=pd.Timestamp("2019-01-05 04:00"),
                       max_depth=2,
                       max_features=100)

        dask_fm, _ = ft.dfs(entityset=dask_es,
                            target_entity=entity.id,
                            trans_primitives=trans_primitives,
                            agg_primitives=agg_primitives,
                            cutoff_time=pd.Timestamp("2019-01-05 04:00"),
                            max_depth=2,
                            max_features=100)
        # Use the same columns and make sure both indexes are sorted the same
        dask_computed_fm = dask_fm.compute().set_index(entity.index).loc[fm.index][fm.columns]
        pd.testing.assert_frame_equal(fm, dask_computed_fm)


def test_aggregation(es, dask_es):
    primitives = ft.list_primitives()
    trans_primitives = []
    agg_list = primitives[primitives['type'] == 'aggregation']['name'].tolist()
    not_supported = ['trend', 'first', 'last', 'time_since_first', 'time_since_last']
    agg_primitives = [prim for prim in agg_list if prim not in not_supported]

    assert es == dask_es

    # Run DFS using each entity as a target and confirm results match
    for entity in es.entities:
        # remove n_most_common for customers due to ambiguity
        if entity.id == 'customers':
            agg_primitives.remove('n_most_common')
        fm, _ = ft.dfs(entityset=es,
                       target_entity=entity.id,
                       trans_primitives=trans_primitives,
                       agg_primitives=agg_primitives,
                       cutoff_time=pd.Timestamp("2019-01-05 04:00"),
                       max_depth=2)

        dask_fm, _ = ft.dfs(entityset=dask_es,
                            target_entity=entity.id,
                            trans_primitives=trans_primitives,
                            agg_primitives=agg_primitives,
                            cutoff_time=pd.Timestamp("2019-01-05 04:00"),
                            max_depth=2)
        if entity.id == 'customers':
            agg_primitives.append('n_most_common')
        # Use the same columns and make sure both indexes are sorted the same
        dask_computed_fm = dask_fm.compute().set_index(entity.index).loc[fm.index][fm.columns]
        pd.testing.assert_frame_equal(fm, dask_computed_fm, check_dtype=False)


def test_create_entity_from_dask_df(es):
    dask_es = EntitySet(id="dask_es")
    log_dask = dd.from_pandas(es["log"].df, npartitions=2)
    dask_es = dask_es.entity_from_dataframe(
        entity_id="log_dask",
        dataframe=log_dask,
        index="id",
        time_index="datetime",
        variable_types=es["log"].variable_types
    )
    pd.testing.assert_frame_equal(es["log"].df, dask_es["log_dask"].df.compute(), check_like=True)


def test_create_entity_with_non_numeric_index(es, dask_es):
    df = pd.DataFrame({"id": ["A_1", "A_2", "C", "D"],
                       "values": [1, 12, -34, 27]})
    dask_df = dd.from_pandas(df, npartitions=2)

    es.entity_from_dataframe(
        entity_id="new_entity",
        dataframe=df,
        index="id")

    dask_es.entity_from_dataframe(
        entity_id="new_entity",
        dataframe=dask_df,
        index="id",
        variable_types={"id": ft.variable_types.Id, "values": ft.variable_types.Numeric})

    pd.testing.assert_frame_equal(es['new_entity'].df.reset_index(drop=True), dask_es['new_entity'].df.compute())


def test_create_entityset_with_mixed_dataframe_types(es, dask_es):
    df = pd.DataFrame({"id": [0, 1, 2, 3],
                       "values": [1, 12, -34, 27]})
    dask_df = dd.from_pandas(df, npartitions=2)

    # Test error is raised when trying to add Dask entity to entitset with existing pandas entities
    err_msg = "All entity dataframes must be of the same type. " \
              "Cannot add entity of type {} to an entityset with existing entities " \
              "of type {}".format(type(dask_df), type(es.entities[0].df))

    with pytest.raises(ValueError, match=err_msg):
        es.entity_from_dataframe(
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
    es = EntitySet(id="es")
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
        "strings": ft.variable_types.Text
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

    es.entity_from_dataframe(entity_id="sessions", dataframe=sessions, index="id", time_index="time")
    dask_es.entity_from_dataframe(entity_id="sessions", dataframe=sessions_dask, index="id", time_index="time", variable_types=sessions_vtypes)

    es.entity_from_dataframe(entity_id="transactions", dataframe=transactions, index="id", time_index="time")
    dask_es.entity_from_dataframe(entity_id="transactions", dataframe=transactions_dask, index="id", time_index="time", variable_types=transactions_vtypes)

    new_rel = Relationship(es["sessions"]["id"],
                           es["transactions"]["session_id"])
    dask_rel = Relationship(dask_es["sessions"]["id"],
                            dask_es["transactions"]["session_id"])

    es = es.add_relationship(new_rel)
    dask_es = dask_es.add_relationship(dask_rel)

    assert es['sessions'].last_time_index is None
    assert dask_es['sessions'].last_time_index is None

    es.add_last_time_indexes()
    dask_es.add_last_time_indexes()

    pd.testing.assert_series_equal(es['sessions'].last_time_index.sort_index(), dask_es['sessions'].last_time_index.compute())


def test_create_entity_with_make_index():
    df = pd.DataFrame({"values": [1, 12, -34, 27]})
    dask_df = dd.from_pandas(df, npartitions=2)
    dask_es = EntitySet(id="dask_es")
    vtypes = {"values": ft.variable_types.Numeric}
    dask_es.entity_from_dataframe(entity_id="new_entity", dataframe=dask_df, make_index=True, index="new_index", variable_types=vtypes)

    assert dask_es['new_entity'].df.columns[0] == "new_index"
    assert dask_es['new_entity'].df['new_index'].compute().to_list() == [0, 1, 2, 3]


def test_single_table_dask_entityset():
    primitives_list = ['cum_sum', 'diff', 'absolute', 'is_weekend', 'year', 'day', 'num_characters', 'num_words']

    dask_es = EntitySet(id="dask_es")
    df = pd.DataFrame({"id": [0, 1, 2, 3],
                       "values": [1, 12, -34, 27],
                       "dates": [pd.to_datetime('2019-01-10'),
                                 pd.to_datetime('2019-02-03'),
                                 pd.to_datetime('2019-01-01'),
                                 pd.to_datetime('2017-08-25')],
                       "strings": ["I am a string",
                                   "23",
                                   "abcdef ghijk",
                                   ""]})
    values_dd = dd.from_pandas(df, npartitions=2)
    vtypes = {
        "id": ft.variable_types.Id,
        "values": ft.variable_types.Numeric,
        "dates": ft.variable_types.Datetime,
        "strings": ft.variable_types.Text
    }
    dask_es.entity_from_dataframe(entity_id="data",
                                  dataframe=values_dd,
                                  index="id",
                                  variable_types=vtypes)

    dask_fm, _ = ft.dfs(entityset=dask_es,
                        target_entity="data",
                        trans_primitives=primitives_list)

    es = ft.EntitySet(id="es")
    es.entity_from_dataframe(entity_id="data",
                             dataframe=df,
                             index="id",
                             variable_types={"strings": ft.variable_types.Text})

    fm, _ = ft.dfs(entityset=es,
                   target_entity="data",
                   trans_primitives=primitives_list)

    # Use the same columns and make sure both indexes are sorted the same
    dask_computed_fm = dask_fm.compute().set_index('id').loc[fm.index][fm.columns]
    pd.testing.assert_frame_equal(fm, dask_computed_fm)


def test_single_table_dask_entityset_ids_not_sorted():
    primitives_list = ['cum_sum', 'diff', 'absolute', 'is_weekend', 'year', 'day', 'num_characters', 'num_words']

    dask_es = EntitySet(id="dask_es")
    df = pd.DataFrame({"id": [2, 0, 1, 3],
                       "values": [1, 12, -34, 27],
                       "dates": [pd.to_datetime('2019-01-10'),
                                 pd.to_datetime('2019-02-03'),
                                 pd.to_datetime('2019-01-01'),
                                 pd.to_datetime('2017-08-25')],
                       "strings": ["I am a string",
                                   "23",
                                   "abcdef ghijk",
                                   ""]})
    values_dd = dd.from_pandas(df, npartitions=2)
    vtypes = {
        "id": ft.variable_types.Id,
        "values": ft.variable_types.Numeric,
        "dates": ft.variable_types.Datetime,
        "strings": ft.variable_types.Text
    }
    dask_es.entity_from_dataframe(entity_id="data",
                                  dataframe=values_dd,
                                  index="id",
                                  variable_types=vtypes)

    dask_fm, _ = ft.dfs(entityset=dask_es,
                        target_entity="data",
                        trans_primitives=primitives_list)

    es = ft.EntitySet(id="es")
    es.entity_from_dataframe(entity_id="data",
                             dataframe=df,
                             index="id",
                             variable_types={"strings": ft.variable_types.Text})

    fm, _ = ft.dfs(entityset=es,
                   target_entity="data",
                   trans_primitives=primitives_list)

    # Make sure both indexes are sorted the same
    pd.testing.assert_frame_equal(fm, dask_fm.compute().set_index('id').loc[fm.index])


def test_single_table_dask_entityset_with_instance_ids():
    primitives_list = ['cum_sum', 'diff', 'absolute', 'is_weekend', 'year', 'day', 'num_characters', 'num_words']
    instance_ids = [0, 1, 3]

    dask_es = EntitySet(id="dask_es")
    df = pd.DataFrame({"id": [0, 1, 2, 3],
                       "values": [1, 12, -34, 27],
                       "dates": [pd.to_datetime('2019-01-10'),
                                 pd.to_datetime('2019-02-03'),
                                 pd.to_datetime('2019-01-01'),
                                 pd.to_datetime('2017-08-25')],
                       "strings": ["I am a string",
                                   "23",
                                   "abcdef ghijk",
                                   ""]})

    values_dd = dd.from_pandas(df, npartitions=2)
    vtypes = {
        "id": ft.variable_types.Id,
        "values": ft.variable_types.Numeric,
        "dates": ft.variable_types.Datetime,
        "strings": ft.variable_types.Text
    }
    dask_es.entity_from_dataframe(entity_id="data",
                                  dataframe=values_dd,
                                  index="id",
                                  variable_types=vtypes)

    dask_fm, _ = ft.dfs(entityset=dask_es,
                        target_entity="data",
                        trans_primitives=primitives_list,
                        instance_ids=instance_ids)

    es = ft.EntitySet(id="es")
    es.entity_from_dataframe(entity_id="data",
                             dataframe=df,
                             index="id",
                             variable_types={"strings": ft.variable_types.Text})

    fm, _ = ft.dfs(entityset=es,
                   target_entity="data",
                   trans_primitives=primitives_list,
                   instance_ids=instance_ids)

    # Make sure both indexes are sorted the same
    pd.testing.assert_frame_equal(fm, dask_fm.compute().set_index('id').loc[fm.index])


def test_single_table_dask_entityset_single_cutoff_time():
    primitives_list = ['cum_sum', 'diff', 'absolute', 'is_weekend', 'year', 'day', 'num_characters', 'num_words']

    dask_es = EntitySet(id="dask_es")
    df = pd.DataFrame({"id": [0, 1, 2, 3],
                       "values": [1, 12, -34, 27],
                       "dates": [pd.to_datetime('2019-01-10'),
                                 pd.to_datetime('2019-02-03'),
                                 pd.to_datetime('2019-01-01'),
                                 pd.to_datetime('2017-08-25')],
                       "strings": ["I am a string",
                                   "23",
                                   "abcdef ghijk",
                                   ""]})
    values_dd = dd.from_pandas(df, npartitions=2)
    vtypes = {
        "id": ft.variable_types.Id,
        "values": ft.variable_types.Numeric,
        "dates": ft.variable_types.Datetime,
        "strings": ft.variable_types.Text
    }
    dask_es.entity_from_dataframe(entity_id="data",
                                  dataframe=values_dd,
                                  index="id",
                                  variable_types=vtypes)

    dask_fm, _ = ft.dfs(entityset=dask_es,
                        target_entity="data",
                        trans_primitives=primitives_list,
                        cutoff_time=pd.Timestamp("2019-01-05 04:00"))

    es = ft.EntitySet(id="es")
    es.entity_from_dataframe(entity_id="data",
                             dataframe=df,
                             index="id",
                             variable_types={"strings": ft.variable_types.Text})

    fm, _ = ft.dfs(entityset=es,
                   target_entity="data",
                   trans_primitives=primitives_list,
                   cutoff_time=pd.Timestamp("2019-01-05 04:00"))

    # Make sure both indexes are sorted the same
    pd.testing.assert_frame_equal(fm, dask_fm.compute().set_index('id').loc[fm.index])


def test_single_table_dask_entityset_cutoff_time_df():
    primitives_list = ['cum_sum', 'diff', 'absolute', 'is_weekend', 'year', 'day', 'num_characters', 'num_words']

    dask_es = EntitySet(id="dask_es")
    df = pd.DataFrame({"id": [0, 1, 2],
                       "values": [1, 12, -34],
                       "dates": [pd.to_datetime('2019-01-10'),
                                 pd.to_datetime('2019-02-03'),
                                 pd.to_datetime('2019-01-01')],
                       "strings": ["I am a string",
                                   "23",
                                   "abcdef ghijk"]})
    values_dd = dd.from_pandas(df, npartitions=2)
    vtypes = {
        "id": ft.variable_types.Id,
        "values": ft.variable_types.Numeric,
        "dates": ft.variable_types.Datetime,
        "strings": ft.variable_types.Text
    }
    dask_es.entity_from_dataframe(entity_id="data",
                                  dataframe=values_dd,
                                  index="id",
                                  variable_types=vtypes)
    ids = [0, 1, 2, 0]
    times = [pd.Timestamp("2019-01-05 04:00"),
             pd.Timestamp("2019-01-05 04:00"),
             pd.Timestamp("2019-01-05 04:00"),
             pd.Timestamp("2019-01-15 04:00")]
    labels = [True, False, True, False]
    cutoff_times = pd.DataFrame({"id": ids, "time": times, "labels": labels}, columns=["id", "time", "labels"])

    dask_fm, _ = ft.dfs(entityset=dask_es,
                        target_entity="data",
                        trans_primitives=primitives_list,
                        cutoff_time=cutoff_times)

    es = ft.EntitySet(id="es")
    es.entity_from_dataframe(entity_id="data",
                             dataframe=df,
                             index="id",
                             variable_types={"strings": ft.variable_types.Text})

    fm, _ = ft.dfs(entityset=es,
                   target_entity="data",
                   trans_primitives=primitives_list,
                   cutoff_time=cutoff_times)

    pd.testing.assert_frame_equal(fm, dask_fm.compute().set_index('id'))


def test_single_table_dask_entityset_dates_not_sorted():
    dask_es = EntitySet(id="dask_es")
    df = pd.DataFrame({"id": [0, 1, 2, 3],
                       "values": [1, 12, -34, 27],
                       "dates": [pd.to_datetime('2019-01-10'),
                                 pd.to_datetime('2019-02-03'),
                                 pd.to_datetime('2019-01-01'),
                                 pd.to_datetime('2017-08-25')]})

    primitives_list = ['cum_sum', 'diff', 'absolute', 'is_weekend', 'year', 'day']
    values_dd = dd.from_pandas(df, npartitions=1)
    vtypes = {
        "id": ft.variable_types.Id,
        "values": ft.variable_types.Numeric,
        "dates": ft.variable_types.Datetime,
    }
    dask_es.entity_from_dataframe(entity_id="data",
                                  dataframe=values_dd,
                                  index="id",
                                  time_index="dates",
                                  variable_types=vtypes)

    dask_fm, _ = ft.dfs(entityset=dask_es,
                        target_entity="data",
                        trans_primitives=primitives_list,
                        max_depth=1)

    es = ft.EntitySet(id="es")
    es.entity_from_dataframe(entity_id="data",
                             dataframe=df,
                             index="id",
                             time_index="dates")

    fm, _ = ft.dfs(entityset=es,
                   target_entity="data",
                   trans_primitives=primitives_list,
                   max_depth=1)

    pd.testing.assert_frame_equal(fm, dask_fm.compute().set_index('id').loc[fm.index])


# def test_training_window_parameter(mock_customer_es, mock_customer_dask_es):
#     entity = "customers"
#     cutoff_times = pd.DataFrame()
#     cutoff_times['customer_id'] = [1, 2, 3, 1]
#     cutoff_times['time'] = pd.to_datetime(['2014-1-1 04:00',
#                                            '2014-1-1 05:00',
#                                            '2014-1-1 06:00',
#                                            '2014-1-1 08:00'])
#     cutoff_times['label'] = [True, True, False, True]

#     cutoff_times_dask = dd.from_pandas(cutoff_times, npartitions=mock_customer_dask_es[entity].df.npartitions)

#     dask_fm, _ = ft.dfs(entityset=mock_customer_dask_es,
#                         target_entity=entity,
#                         cutoff_time=cutoff_times_dask,
#                         training_window="2 hour")

#     fm, _ = ft.dfs(entityset=mock_customer_es,
#                    target_entity=entity,
#                    cutoff_time=cutoff_times,
#                    training_window="2 hour")

#     pd.testing.assert_frame_equal(fm, dask_fm.compute().set_index('customer_id'))


def test_secondary_time_index():
    log_df = pd.DataFrame()
    log_df['id'] = [0, 1, 2, 3]
    log_df['scheduled_time'] = pd.to_datetime([
        "2019-01-01",
        "2019-01-01",
        "2019-01-01",
        "2019-01-01"
    ])
    log_df['departure_time'] = pd.to_datetime([
        "2019-02-01 09:00",
        "2019-02-06 10:00",
        "2019-02-12 10:00",
        "2019-03-01 11:30"])
    log_df['arrival_time'] = pd.to_datetime([
        "2019-02-01 11:23",
        "2019-02-06 12:45",
        "2019-02-12 13:53",
        "2019-03-01 14:07"
    ])
    log_df['delay'] = [-2, 10, 60, 0]
    log_df['flight_id'] = [0, 1, 0, 1]
    log_dask = dd.from_pandas(log_df, npartitions=2)

    flights_df = pd.DataFrame()
    flights_df['id'] = [0, 1, 2, 3]
    flights_df['origin'] = ["BOS", "LAX", "BOS", "LAX"]
    flights_dask = dd.from_pandas(flights_df, npartitions=2)

    es = ft.EntitySet("flights")
    dask_es = ft.EntitySet("flights_dask")

    es.entity_from_dataframe(entity_id='logs',
                             dataframe=log_df,
                             index="id",
                             time_index="scheduled_time",
                             secondary_time_index={
                                 'arrival_time': ['departure_time', 'delay']})

    log_vtypes = {
        "id": ft.variable_types.Id,
        "scheduled_time": ft.variable_types.DatetimeTimeIndex,
        "departure_time": ft.variable_types.DatetimeTimeIndex,
        "arrival_time": ft.variable_types.DatetimeTimeIndex,
        "delay": ft.variable_types.Numeric,
        "flight_id": ft.variable_types.Id
    }
    dask_es.entity_from_dataframe(entity_id='logs',
                                  dataframe=log_dask,
                                  index="id",
                                  variable_types=log_vtypes,
                                  time_index="scheduled_time",
                                  secondary_time_index={
                                      'arrival_time': ['departure_time', 'delay']})

    es.entity_from_dataframe('flights', flights_df, index="id")
    flights_vtypes = es['flights'].variable_types
    dask_es.entity_from_dataframe('flights', flights_dask, index="id", variable_types=flights_vtypes)

    new_rel = ft.Relationship(es['flights']['id'], es['logs']['flight_id'])
    dask_rel = ft.Relationship(dask_es['flights']['id'], dask_es['logs']['flight_id'])
    es.add_relationship(new_rel)
    dask_es.add_relationship(dask_rel)

    cutoff_df = pd.DataFrame()
    cutoff_df['id'] = [0, 1, 1]
    cutoff_df['time'] = pd.to_datetime(['2019-02-02', '2019-02-02', '2019-02-20'])

    fm, _ = ft.dfs(entityset=es,
                   target_entity="logs",
                   cutoff_time=cutoff_df,
                   agg_primitives=["max"],
                   trans_primitives=["month"])

    dask_fm, _ = ft.dfs(entityset=dask_es,
                        target_entity="logs",
                        cutoff_time=cutoff_df,
                        agg_primitives=["max"],
                        trans_primitives=["month"])

    # Make sure both matrixes are sorted the same
    pd.testing.assert_frame_equal(fm.sort_values('delay'), dask_fm.compute().set_index('id').sort_values('delay'))


def test_build_es_from_scratch_and_run_dfs():
    es = ft.demo.load_mock_customer(return_entityset=True)
    data = ft.demo.load_mock_customer()

    transactions_df = data["transactions"].merge(data["sessions"]).merge(data["customers"])
    transactions_dd = dd.from_pandas(transactions_df, npartitions=4)
    products_dd = dd.from_pandas(data["products"], npartitions=4)
    dask_es = EntitySet(id="transactions")

    transactions_vtypes = {
        "transaction_id": ft.variable_types.Id,
        "session_id": ft.variable_types.Id,
        "transaction_time": ft.variable_types.DatetimeTimeIndex,
        "product_id": ft.variable_types.Id,
        "amount": ft.variable_types.Numeric,
        "customer_id": ft.variable_types.Id,
        "device": ft.variable_types.Categorical,
        "session_start": ft.variable_types.DatetimeTimeIndex,
        "zip_code": ft.variable_types.ZIPCode,
        "join_date": ft.variable_types.Datetime,
        "date_of_birth": ft.variable_types.Datetime
    }
    dask_es.entity_from_dataframe(entity_id="transactions",
                                  dataframe=transactions_dd,
                                  index="transaction_id",
                                  time_index="transaction_time",
                                  variable_types=transactions_vtypes)

    products_vtypes = {
        "product_id": ft.variable_types.Id,
        "brand": ft.variable_types.Categorical
    }
    dask_es.entity_from_dataframe(entity_id="products",
                                  dataframe=products_dd,
                                  index="product_id",
                                  variable_types=products_vtypes)

    new_rel = Relationship(dask_es["products"]["product_id"],
                           dask_es["transactions"]["product_id"])

    dask_es = dask_es.add_relationship(new_rel)

    dask_es = dask_es.normalize_entity(base_entity_id="transactions",
                                       new_entity_id="sessions",
                                       index="session_id",
                                       make_time_index="session_start",
                                       additional_variables=["device",
                                                             "customer_id",
                                                             "zip_code",
                                                             "session_start",
                                                             "join_date",
                                                             "date_of_birth"])

    dask_es = dask_es.normalize_entity(base_entity_id="sessions",
                                       new_entity_id="customers",
                                       index="customer_id",
                                       make_time_index="join_date",
                                       additional_variables=["zip_code", "join_date", "date_of_birth"])

    trans_primitives = ['cum_sum', 'diff', 'absolute', 'is_weekend', 'year', 'day', 'num_characters', 'num_words']
    agg_primitives = ['num_unique', 'count', 'max', 'sum']

    fm, _ = ft.dfs(entityset=es,
                   target_entity="customers",
                   trans_primitives=trans_primitives,
                   agg_primitives=agg_primitives,
                   max_depth=2)

    dask_fm, _ = ft.dfs(entityset=dask_es,
                        target_entity="customers",
                        trans_primitives=trans_primitives,
                        agg_primitives=agg_primitives,
                        max_depth=2)

    # Use the same columns and make sure both have same index sorting
    breakpoint()
    pd.testing.assert_frame_equal(fm, dask_fm.compute().set_index('customer_id')[fm.columns], check_dtype=False)
