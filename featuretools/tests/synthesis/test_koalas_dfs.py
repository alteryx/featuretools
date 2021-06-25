import pandas as pd
import pytest
import woodwork as ww

import featuretools as ft
from featuretools.entityset import EntitySet
from featuretools.utils.gen_utils import import_or_none

ks = import_or_none('databricks.koalas')


@pytest.mark.skipif('not ks')
def test_single_table_ks_entityset():
    primitives_list = ['absolute', 'is_weekend', 'year', 'day', 'num_characters', 'num_words']

    ks_es = EntitySet(id="ks_es")
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
    values_dd = ks.from_pandas(df)
    ltypes = {
        "values": ww.logical_types.Double,
        "dates": ww.logical_types.Datetime,
        "strings": ww.logical_types.NaturalLanguage
    }
    ks_es.add_dataframe(
        dataframe_name="data",
        dataframe=values_dd,
        index="id",
        logical_types=ltypes)

    ks_fm, _ = ft.dfs(entityset=ks_es,
                      target_entity="data",
                      trans_primitives=primitives_list)

    pd_es = ft.EntitySet(id="pd_es")
    pd_es.add_dataframe(
        dataframe_name="data",
        dataframe=df,
        index="id",
        logical_types={"strings": ww.logical_types.NaturalLanguage})

    fm, _ = ft.dfs(entityset=pd_es,
                   target_entity="data",
                   trans_primitives=primitives_list)

    ks_computed_fm = ks_fm.to_pandas().set_index('id').loc[fm.index][fm.columns]
    # NUM_WORDS(strings) is int32 in koalas for some reason
    pd.testing.assert_frame_equal(fm, ks_computed_fm, check_dtype=False)


@pytest.mark.skipif('not ks')
def test_single_table_ks_entityset_ids_not_sorted():
    primitives_list = ['absolute', 'is_weekend', 'year', 'day', 'num_characters', 'num_words']

    ks_es = EntitySet(id="ks_es")
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
    values_dd = ks.from_pandas(df)
    ltypes = {
        "values": ww.logical_types.Double,
        "dates": ww.logical_types.Datetime,
        "strings": ww.logical_types.NaturalLanguage,
    }
    ks_es.add_dataframe(
        dataframe_name="data",
        dataframe=values_dd,
        index="id",
        logical_types=ltypes)

    ks_fm, _ = ft.dfs(entityset=ks_es,
                      target_entity="data",
                      trans_primitives=primitives_list)

    pd_es = ft.EntitySet(id="pd_es")
    pd_es.add_dataframe(
        dataframe_name="data",
        dataframe=df,
        index="id",
        logical_types={"strings": ww.logical_types.NaturalLanguage})

    fm, _ = ft.dfs(entityset=pd_es,
                   target_entity="data",
                   trans_primitives=primitives_list)

    # Make sure both indexes are sorted the same
    pd.testing.assert_frame_equal(fm, ks_fm.to_pandas().set_index('id').loc[fm.index], check_dtype=False)


@pytest.mark.skipif('not ks')
def test_single_table_ks_entityset_with_instance_ids():
    primitives_list = ['absolute', 'is_weekend', 'year', 'day', 'num_characters', 'num_words']
    instance_ids = [0, 1, 3]

    ks_es = EntitySet(id="ks_es")
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

    values_dd = ks.from_pandas(df)
    ltypes = {
        "values": ww.logical_types.Double,
        "dates": ww.logical_types.Datetime,
        "strings": ww.logical_types.NaturalLanguage
    }
    ks_es.add_dataframe(
        dataframe_name="data",
        dataframe=values_dd,
        index="id",
        logical_types=ltypes)

    ks_fm, _ = ft.dfs(entityset=ks_es,
                      target_entity="data",
                      trans_primitives=primitives_list,
                      instance_ids=instance_ids)

    pd_es = ft.EntitySet(id="pd_es")
    pd_es.add_dataframe(
        dataframe_name="data",
        dataframe=df,
        index="id",
        logical_types={"strings": ww.logical_types.NaturalLanguage})

    fm, _ = ft.dfs(entityset=pd_es,
                   target_entity="data",
                   trans_primitives=primitives_list,
                   instance_ids=instance_ids)

    # Make sure both indexes are sorted the same
    pd.testing.assert_frame_equal(fm, ks_fm.to_pandas().set_index('id').loc[fm.index], check_dtype=False)


@pytest.mark.skipif('not ks')
def test_single_table_ks_entityset_single_cutoff_time():
    primitives_list = ['absolute', 'is_weekend', 'year', 'day', 'num_characters', 'num_words']

    ks_es = EntitySet(id="ks_es")
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
    values_dd = ks.from_pandas(df)
    ltypes = {
        "values": ww.logical_types.Double,
        "dates": ww.logical_types.Datetime,
        "strings": ww.logical_types.NaturalLanguage
    }
    ks_es.add_dataframe(
        dataframe_name="data",
        dataframe=values_dd,
        index="id",
        logical_types=ltypes)

    ks_fm, _ = ft.dfs(entityset=ks_es,
                      target_entity="data",
                      trans_primitives=primitives_list,
                      cutoff_time=pd.Timestamp("2019-01-05 04:00"))

    pd_es = ft.EntitySet(id="pd_es")
    pd_es.add_dataframe(
        dataframe_name="data",
        dataframe=df,
        index="id",
        logical_types={"strings": ww.logical_types.NaturalLanguage})

    fm, _ = ft.dfs(entityset=pd_es,
                   target_entity="data",
                   trans_primitives=primitives_list,
                   cutoff_time=pd.Timestamp("2019-01-05 04:00"))

    # Make sure both indexes are sorted the same
    pd.testing.assert_frame_equal(fm, ks_fm.to_pandas().set_index('id').loc[fm.index], check_dtype=False)


@pytest.mark.skipif('not ks')
def test_single_table_ks_entityset_cutoff_time_df():
    primitives_list = ['absolute', 'is_weekend', 'year', 'day', 'num_characters', 'num_words']

    ks_es = EntitySet(id="ks_es")
    df = pd.DataFrame({"id": [0, 1, 2],
                       "values": [1, 12, -34],
                       "dates": [pd.to_datetime('2019-01-10'),
                                 pd.to_datetime('2019-02-03'),
                                 pd.to_datetime('2019-01-01')],
                       "strings": ["I am a string",
                                   "23",
                                   "abcdef ghijk"]})
    values_dd = ks.from_pandas(df)
    ltypes = {
        "values": ww.logical_types.Numeric,
        "dates": ww.logical_types.Datetime,
        "strings": ww.logical_types.NaturalLanguage
    }
    ks_es.add_dataframe(
        dataframe_name="data",
        dataframe=values_dd,
        index="id",
        time_index="dates",
        logical_types=ltypes)

    ids = [0, 1, 2, 0]
    times = [pd.Timestamp("2019-01-05 04:00"),
             pd.Timestamp("2019-01-05 04:00"),
             pd.Timestamp("2019-01-05 04:00"),
             pd.Timestamp("2019-01-15 04:00")]
    labels = [True, False, True, False]
    cutoff_times = pd.DataFrame({"id": ids, "time": times, "labels": labels}, columns=["id", "time", "labels"])

    ks_fm, _ = ft.dfs(entityset=ks_es,
                      target_entity="data",
                      trans_primitives=primitives_list,
                      cutoff_time=cutoff_times)

    pd_es = ft.EntitySet(id="pd_es")
    pd_es.add_dataframe(
        dataframe_name="data",
        dataframe=df,
        index="id",
        time_index="dates",
        logical_types={"strings": ww.logical_types.NaturalLanguage})

    fm, _ = ft.dfs(entityset=pd_es,
                   target_entity="data",
                   trans_primitives=primitives_list,
                   cutoff_time=cutoff_times)
    # Because row ordering with koalas is not guaranteed, `we need to sort on two columns to make sure that values
    # for instance id 0 are compared correctly. Also, make sure the boolean column has the same dtype.
    fm = fm.sort_values(['id', 'labels'])
    ks_fm = ks_fm.to_pandas().set_index('id').sort_values(['id', 'labels'])
    ks_fm['IS_WEEKEND(dates)'] = ks_fm['IS_WEEKEND(dates)'].astype(fm['IS_WEEKEND(dates)'].dtype)
    pd.testing.assert_frame_equal(fm, ks_fm)


@pytest.mark.skipif('not ks')
def test_single_table_ks_entityset_dates_not_sorted():
    ks_es = EntitySet(id="ks_es")
    df = pd.DataFrame({"id": [0, 1, 2, 3],
                       "values": [1, 12, -34, 27],
                       "dates": [pd.to_datetime('2019-01-10'),
                                 pd.to_datetime('2019-02-03'),
                                 pd.to_datetime('2019-01-01'),
                                 pd.to_datetime('2017-08-25')]})

    primitives_list = ['absolute', 'is_weekend', 'year', 'day']
    values_dd = ks.from_pandas(df)
    ltypes = {
        "values": ww.logical_types.Double,
        "dates": ww.logical_types.Datetime,
    }
    ks_es.add_dataframe(
        dataframe_name="data",
        dataframe=values_dd,
        index="id",
        time_index="dates",
        logical_types=ltypes)

    ks_fm, _ = ft.dfs(entityset=ks_es,
                      target_entity="data",
                      trans_primitives=primitives_list,
                      max_depth=1)

    pd_es = ft.EntitySet(id="pd_es")
    pd_es.add_dataframe(
        dataframe_name="data",
        dataframe=df,
        index="id",
        time_index="dates")

    fm, _ = ft.dfs(entityset=pd_es,
                   target_entity="data",
                   trans_primitives=primitives_list,
                   max_depth=1)

    pd.testing.assert_frame_equal(fm, ks_fm.to_pandas().set_index('id').loc[fm.index])


@pytest.mark.skipif('not ks')
def test_ks_entityset_secondary_time_index():
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
    log_ks = ks.from_pandas(log_df)

    flights_df = pd.DataFrame()
    flights_df['id'] = [0, 1, 2, 3]
    flights_df['origin'] = ["BOS", "LAX", "BOS", "LAX"]
    flights_ks = ks.from_pandas(flights_df)

    pd_es = ft.EntitySet("flights")
    ks_es = ft.EntitySet("flights_ks")

    pd_es.add_dataframe(
        dataframe_name='logs',
        dataframe=log_df,
        index="id",
        time_index="scheduled_time",
        secondary_time_index={'arrival_time': ['departure_time', 'delay']})

    log_ltypes = {
        "scheduled_time": ww.logical_types.Datetime,
        "departure_time": ww.logical_types.Datetime,
        "arrival_time": ww.logical_types.Datetime,
        "delay": ww.logical_types.Double,
    }
    ks_es.add_dataframe(
        dataframe_name='logs',
        dataframe=log_ks,
        index="id",
        logical_types=log_ltypes,
        semantic_tags={'flight_id': 'foreign_key'},
        time_index="scheduled_time",
        secondary_time_index={'arrival_time': ['departure_time', 'delay']})

    pd_es.add_dataframe(dataframe_name='flights', dataframe=flights_df, index="id")
    flights_ltypes = pd_es['flights'].variable_types
    ks_es.add_dataframe(dataframe_name='flights', dataframe=flights_ks, index="id", logical_types=flights_ltypes)

    pd_es.add_relationship('flights', 'id', 'logs', 'flight_id')
    ks_es.add_relationship('flights', 'id', 'logs', 'flight_id')

    cutoff_df = pd.DataFrame()
    cutoff_df['id'] = [0, 1, 1]
    cutoff_df['time'] = pd.to_datetime(['2019-02-02', '2019-02-02', '2019-02-20'])

    fm, _ = ft.dfs(entityset=pd_es,
                   target_entity="logs",
                   cutoff_time=cutoff_df,
                   agg_primitives=["max"],
                   trans_primitives=["month"])

    ks_fm, _ = ft.dfs(entityset=ks_es,
                      target_entity="logs",
                      cutoff_time=cutoff_df,
                      agg_primitives=["max"],
                      trans_primitives=["month"])

    # Make sure both matrixes are sorted the same
    pd.testing.assert_frame_equal(fm.sort_values('delay'), ks_fm.to_pandas().set_index('id').sort_values('delay'), check_dtype=False)
