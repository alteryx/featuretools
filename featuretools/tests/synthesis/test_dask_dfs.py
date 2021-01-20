import dask.dataframe as dd
import pandas as pd

import featuretools as ft
from featuretools.entityset import EntitySet


def test_single_table_dask_entityset():
    primitives_list = ['absolute', 'is_weekend', 'year', 'day', 'num_characters', 'num_words']

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
        "strings": ft.variable_types.NaturalLanguage
    }
    dask_es.entity_from_dataframe(entity_id="data",
                                  dataframe=values_dd,
                                  index="id",
                                  variable_types=vtypes)

    dask_fm, _ = ft.dfs(entityset=dask_es,
                        target_entity="data",
                        trans_primitives=primitives_list)

    pd_es = ft.EntitySet(id="pd_es")
    pd_es.entity_from_dataframe(entity_id="data",
                                dataframe=df,
                                index="id",
                                variable_types={"strings": ft.variable_types.NaturalLanguage})

    fm, _ = ft.dfs(entityset=pd_es,
                   target_entity="data",
                   trans_primitives=primitives_list)

    # Use the same columns and make sure both indexes are sorted the same
    dask_computed_fm = dask_fm.compute().set_index('id').loc[fm.index][fm.columns]
    pd.testing.assert_frame_equal(fm, dask_computed_fm)


def test_single_table_dask_entityset_ids_not_sorted():
    primitives_list = ['absolute', 'is_weekend', 'year', 'day', 'num_characters', 'num_words']

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
        "strings": ft.variable_types.NaturalLanguage
    }
    dask_es.entity_from_dataframe(entity_id="data",
                                  dataframe=values_dd,
                                  index="id",
                                  variable_types=vtypes)

    dask_fm, _ = ft.dfs(entityset=dask_es,
                        target_entity="data",
                        trans_primitives=primitives_list)

    pd_es = ft.EntitySet(id="pd_es")
    pd_es.entity_from_dataframe(entity_id="data",
                                dataframe=df,
                                index="id",
                                variable_types={"strings": ft.variable_types.NaturalLanguage})

    fm, _ = ft.dfs(entityset=pd_es,
                   target_entity="data",
                   trans_primitives=primitives_list)

    # Make sure both indexes are sorted the same
    pd.testing.assert_frame_equal(fm, dask_fm.compute().set_index('id').loc[fm.index])


def test_single_table_dask_entityset_with_instance_ids():
    primitives_list = ['absolute', 'is_weekend', 'year', 'day', 'num_characters', 'num_words']
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
        "strings": ft.variable_types.NaturalLanguage
    }
    dask_es.entity_from_dataframe(entity_id="data",
                                  dataframe=values_dd,
                                  index="id",
                                  variable_types=vtypes)

    dask_fm, _ = ft.dfs(entityset=dask_es,
                        target_entity="data",
                        trans_primitives=primitives_list,
                        instance_ids=instance_ids)

    pd_es = ft.EntitySet(id="pd_es")
    pd_es.entity_from_dataframe(entity_id="data",
                                dataframe=df,
                                index="id",
                                variable_types={"strings": ft.variable_types.NaturalLanguage})

    fm, _ = ft.dfs(entityset=pd_es,
                   target_entity="data",
                   trans_primitives=primitives_list,
                   instance_ids=instance_ids)

    # Make sure both indexes are sorted the same
    pd.testing.assert_frame_equal(fm, dask_fm.compute().set_index('id').loc[fm.index])


def test_single_table_dask_entityset_single_cutoff_time():
    primitives_list = ['absolute', 'is_weekend', 'year', 'day', 'num_characters', 'num_words']

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
        "strings": ft.variable_types.NaturalLanguage
    }
    dask_es.entity_from_dataframe(entity_id="data",
                                  dataframe=values_dd,
                                  index="id",
                                  variable_types=vtypes)

    dask_fm, _ = ft.dfs(entityset=dask_es,
                        target_entity="data",
                        trans_primitives=primitives_list,
                        cutoff_time=pd.Timestamp("2019-01-05 04:00"))

    pd_es = ft.EntitySet(id="pd_es")
    pd_es.entity_from_dataframe(entity_id="data",
                                dataframe=df,
                                index="id",
                                variable_types={"strings": ft.variable_types.NaturalLanguage})

    fm, _ = ft.dfs(entityset=pd_es,
                   target_entity="data",
                   trans_primitives=primitives_list,
                   cutoff_time=pd.Timestamp("2019-01-05 04:00"))

    # Make sure both indexes are sorted the same
    pd.testing.assert_frame_equal(fm, dask_fm.compute().set_index('id').loc[fm.index])


def test_single_table_dask_entityset_cutoff_time_df():
    primitives_list = ['absolute', 'is_weekend', 'year', 'day', 'num_characters', 'num_words']

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
        "dates": ft.variable_types.DatetimeTimeIndex,
        "strings": ft.variable_types.NaturalLanguage
    }
    dask_es.entity_from_dataframe(entity_id="data",
                                  dataframe=values_dd,
                                  index="id",
                                  time_index="dates",
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

    pd_es = ft.EntitySet(id="pd_es")
    pd_es.entity_from_dataframe(entity_id="data",
                                dataframe=df,
                                index="id",
                                time_index="dates",
                                variable_types={"strings": ft.variable_types.NaturalLanguage})

    fm, _ = ft.dfs(entityset=pd_es,
                   target_entity="data",
                   trans_primitives=primitives_list,
                   cutoff_time=cutoff_times)
    # Because row ordering with Dask is not guaranteed, we need to sort on two columns to make sure that values
    # for instance id 0 are compared correctly. Also, make sure the boolean column has the same dtype.
    fm = fm.sort_values(['id', 'labels'])
    dask_fm = dask_fm.compute().set_index('id').sort_values(['id', 'labels'])
    dask_fm['IS_WEEKEND(dates)'] = dask_fm['IS_WEEKEND(dates)'].astype(fm['IS_WEEKEND(dates)'].dtype)
    pd.testing.assert_frame_equal(fm, dask_fm)


def test_single_table_dask_entityset_dates_not_sorted():
    dask_es = EntitySet(id="dask_es")
    df = pd.DataFrame({"id": [0, 1, 2, 3],
                       "values": [1, 12, -34, 27],
                       "dates": [pd.to_datetime('2019-01-10'),
                                 pd.to_datetime('2019-02-03'),
                                 pd.to_datetime('2019-01-01'),
                                 pd.to_datetime('2017-08-25')]})

    primitives_list = ['absolute', 'is_weekend', 'year', 'day']
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

    pd_es = ft.EntitySet(id="pd_es")
    pd_es.entity_from_dataframe(entity_id="data",
                                dataframe=df,
                                index="id",
                                time_index="dates")

    fm, _ = ft.dfs(entityset=pd_es,
                   target_entity="data",
                   trans_primitives=primitives_list,
                   max_depth=1)

    pd.testing.assert_frame_equal(fm, dask_fm.compute().set_index('id').loc[fm.index])


def test_dask_entityset_secondary_time_index():
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

    pd_es = ft.EntitySet("flights")
    dask_es = ft.EntitySet("flights_dask")

    pd_es.entity_from_dataframe(entity_id='logs',
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

    pd_es.entity_from_dataframe('flights', flights_df, index="id")
    flights_vtypes = pd_es['flights'].variable_types
    dask_es.entity_from_dataframe('flights', flights_dask, index="id", variable_types=flights_vtypes)

    new_rel = ft.Relationship(pd_es['flights']['id'], pd_es['logs']['flight_id'])
    dask_rel = ft.Relationship(dask_es['flights']['id'], dask_es['logs']['flight_id'])
    pd_es.add_relationship(new_rel)
    dask_es.add_relationship(dask_rel)

    cutoff_df = pd.DataFrame()
    cutoff_df['id'] = [0, 1, 1]
    cutoff_df['time'] = pd.to_datetime(['2019-02-02', '2019-02-02', '2019-02-20'])

    fm, _ = ft.dfs(entityset=pd_es,
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
