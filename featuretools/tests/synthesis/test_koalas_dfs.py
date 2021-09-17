import pandas as pd
import pytest
from woodwork.logical_types import (
    Datetime,
    Double,
    Integer,
    IntegerNullable,
    NaturalLanguage
)

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
        "values": Integer,
        "dates": Datetime,
        "strings": NaturalLanguage
    }
    ks_es.add_dataframe(
        dataframe_name="data",
        dataframe=values_dd,
        index="id",
        logical_types=ltypes)

    ks_fm, _ = ft.dfs(entityset=ks_es,
                      target_dataframe_name="data",
                      trans_primitives=primitives_list)

    pd_es = ft.EntitySet(id="pd_es")
    pd_es.add_dataframe(
        dataframe_name="data",
        dataframe=df,
        index="id",
        logical_types=ltypes)

    fm, _ = ft.dfs(entityset=pd_es,
                   target_dataframe_name="data",
                   trans_primitives=primitives_list)

    ks_computed_fm = ks_fm.to_pandas().set_index('id').loc[fm.index][fm.columns]
    # Koalas dtypes are different for categorical - set the pandas fm to have the same dtypes before comparing
    pd.testing.assert_frame_equal(fm.astype(ks_computed_fm.dtypes), ks_computed_fm)


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
        "values": Integer,
        "dates": Datetime,
        "strings": NaturalLanguage,
    }
    ks_es.add_dataframe(
        dataframe_name="data",
        dataframe=values_dd,
        index="id",
        logical_types=ltypes)

    ks_fm, _ = ft.dfs(entityset=ks_es,
                      target_dataframe_name="data",
                      trans_primitives=primitives_list)

    pd_es = ft.EntitySet(id="pd_es")
    pd_es.add_dataframe(
        dataframe_name="data",
        dataframe=df,
        index="id",
        logical_types=ltypes)

    fm, _ = ft.dfs(entityset=pd_es,
                   target_dataframe_name="data",
                   trans_primitives=primitives_list)

    ks_computed_fm = ks_fm.to_pandas().set_index('id').loc[fm.index]
    # Koalas dtypes are different for categorical - set the pandas fm to have the same dtypes before comparing
    pd.testing.assert_frame_equal(fm.astype(ks_computed_fm.dtypes), ks_computed_fm)


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
        "values": Integer,
        "dates": Datetime,
        "strings": NaturalLanguage
    }
    ks_es.add_dataframe(
        dataframe_name="data",
        dataframe=values_dd,
        index="id",
        logical_types=ltypes)

    ks_fm, _ = ft.dfs(entityset=ks_es,
                      target_dataframe_name="data",
                      trans_primitives=primitives_list,
                      instance_ids=instance_ids)

    pd_es = ft.EntitySet(id="pd_es")
    pd_es.add_dataframe(
        dataframe_name="data",
        dataframe=df,
        index="id",
        logical_types=ltypes)

    fm, _ = ft.dfs(entityset=pd_es,
                   target_dataframe_name="data",
                   trans_primitives=primitives_list,
                   instance_ids=instance_ids)

    ks_computed_fm = ks_fm.to_pandas().set_index('id').loc[fm.index]
    # Koalas dtypes are different for categorical - set the pandas fm to have the same dtypes before comparing
    pd.testing.assert_frame_equal(fm.astype(ks_computed_fm.dtypes), ks_computed_fm)


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
        "values": Integer,
        "dates": Datetime,
        "strings": NaturalLanguage
    }
    ks_es.add_dataframe(
        dataframe_name="data",
        dataframe=values_dd,
        index="id",
        logical_types=ltypes)

    ks_fm, _ = ft.dfs(entityset=ks_es,
                      target_dataframe_name="data",
                      trans_primitives=primitives_list,
                      cutoff_time=pd.Timestamp("2019-01-05 04:00"))

    pd_es = ft.EntitySet(id="pd_es")
    pd_es.add_dataframe(
        dataframe_name="data",
        dataframe=df,
        index="id",
        logical_types=ltypes)

    fm, _ = ft.dfs(entityset=pd_es,
                   target_dataframe_name="data",
                   trans_primitives=primitives_list,
                   cutoff_time=pd.Timestamp("2019-01-05 04:00"))

    ks_computed_fm = ks_fm.to_pandas().set_index('id').loc[fm.index]
    # Koalas dtypes are different for categorical - set the pandas fm to have the same dtypes before comparing
    pd.testing.assert_frame_equal(fm.astype(ks_computed_fm.dtypes), ks_computed_fm)


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
        "values": IntegerNullable,
        "dates": Datetime,
        "strings": NaturalLanguage
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
                      target_dataframe_name="data",
                      trans_primitives=primitives_list,
                      cutoff_time=cutoff_times)

    pd_es = ft.EntitySet(id="pd_es")
    pd_es.add_dataframe(
        dataframe_name="data",
        dataframe=df,
        index="id",
        time_index="dates",
        logical_types=ltypes)

    fm, _ = ft.dfs(entityset=pd_es,
                   target_dataframe_name="data",
                   trans_primitives=primitives_list,
                   cutoff_time=cutoff_times)
    # Because row ordering with koalas is not guaranteed, `we need to sort on two columns to make sure that values
    # for instance id 0 are compared correctly. Also, make sure the boolean columns have the same dtype.
    fm = fm.sort_values(['id', 'labels'])
    ks_fm = ks_fm.to_pandas().set_index('id').sort_values(['id', 'labels'])

    for column in fm.columns:
        if fm[column].dtype.name == 'category':
            fm[column] = fm[column].astype('Int64').astype('string')

    pd.testing.assert_frame_equal(fm.astype(ks_fm.dtypes), ks_fm)


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
        "values": Integer,
        "dates": Datetime,
    }
    ks_es.add_dataframe(
        dataframe_name="data",
        dataframe=values_dd,
        index="id",
        time_index="dates",
        logical_types=ltypes)

    ks_fm, _ = ft.dfs(entityset=ks_es,
                      target_dataframe_name="data",
                      trans_primitives=primitives_list,
                      max_depth=1)

    pd_es = ft.EntitySet(id="pd_es")
    pd_es.add_dataframe(
        dataframe_name="data",
        dataframe=df,
        index="id",
        time_index="dates",
        logical_types=ltypes)

    fm, _ = ft.dfs(entityset=pd_es,
                   target_dataframe_name="data",
                   trans_primitives=primitives_list,
                   max_depth=1)

    ks_fm = ks_fm.to_pandas().set_index('id').loc[fm.index]
    pd.testing.assert_frame_equal(fm.astype(ks_fm.dtypes), ks_fm)


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

    log_ltypes = {
        "scheduled_time": Datetime,
        "departure_time": Datetime,
        "arrival_time": Datetime,
        "delay": Double,
    }
    pd_es.add_dataframe(
        dataframe_name='logs',
        dataframe=log_df,
        index="id",
        logical_types=log_ltypes,
        semantic_tags={'flight_id': 'foreign_key'},
        time_index="scheduled_time",
        secondary_time_index={'arrival_time': ['departure_time', 'delay']})

    ks_es.add_dataframe(
        dataframe_name='logs',
        dataframe=log_ks,
        index="id",
        logical_types=log_ltypes,
        semantic_tags={'flight_id': 'foreign_key'},
        time_index="scheduled_time",
        secondary_time_index={'arrival_time': ['departure_time', 'delay']})

    pd_es.add_dataframe(dataframe_name='flights', dataframe=flights_df, index="id")
    flights_ltypes = pd_es['flights'].ww.logical_types
    ks_es.add_dataframe(dataframe_name='flights', dataframe=flights_ks, index="id", logical_types=flights_ltypes)

    pd_es.add_relationship('flights', 'id', 'logs', 'flight_id')
    ks_es.add_relationship('flights', 'id', 'logs', 'flight_id')

    cutoff_df = pd.DataFrame()
    cutoff_df['id'] = [0, 1, 1]
    cutoff_df['time'] = pd.to_datetime(['2019-02-02', '2019-02-02', '2019-02-20'])

    fm, _ = ft.dfs(entityset=pd_es,
                   target_dataframe_name="logs",
                   cutoff_time=cutoff_df,
                   agg_primitives=["max"],
                   trans_primitives=["month"])

    ks_fm, _ = ft.dfs(entityset=ks_es,
                      target_dataframe_name="logs",
                      cutoff_time=cutoff_df,
                      agg_primitives=["max"],
                      trans_primitives=["month"])

    # Make sure both matrices are sorted the same
    ks_fm = ks_fm.to_pandas().set_index('id').sort_values('delay')
    fm = fm.sort_values('delay')

    # Koalas output for MONTH columns will be of string type without decimal points,
    # while pandas will contain decimals - we need to convert before comparing
    for column in fm.columns:
        if fm[column].dtype.name == 'category':
            fm[column] = fm[column].astype('Int64').astype('string')

    pd.testing.assert_frame_equal(fm, ks_fm, check_categorical=False)
