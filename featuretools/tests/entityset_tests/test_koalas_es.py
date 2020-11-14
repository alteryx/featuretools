import pandas as pd
import pytest

import featuretools as ft
from featuretools.entityset import EntitySet, Relationship
from featuretools.utils.gen_utils import import_or_none
from featuretools.utils.koalas_utils import pd_to_ks_clean

ks = import_or_none('databricks.koalas')


@pytest.mark.skipif('not ks')
def test_create_entity_from_ks_df(pd_es):
    cleaned_df = pd_to_ks_clean(pd_es["log"].df)
    log_ks = ks.from_pandas(cleaned_df)

    ks_es = EntitySet(id="ks_es")
    ks_es = ks_es.entity_from_dataframe(
        entity_id="log_ks",
        dataframe=log_ks,
        index="id",
        time_index="datetime",
        variable_types=pd_es["log"].variable_types
    )
    pd.testing.assert_frame_equal(cleaned_df, ks_es["log_ks"].df.to_pandas(), check_like=True)


@pytest.mark.skipif('not ks')
def test_create_entity_with_non_numeric_index(pd_es, ks_es):
    df = pd.DataFrame({"id": ["A_1", "A_2", "C", "D"],
                       "values": [1, 12, -34, 27]})
    ks_df = ks.from_pandas(df)

    pd_es.entity_from_dataframe(
        entity_id="new_entity",
        dataframe=df,
        index="id")

    ks_es.entity_from_dataframe(
        entity_id="new_entity",
        dataframe=ks_df,
        index="id",
        variable_types={"id": ft.variable_types.Id, "values": ft.variable_types.Numeric})
    pd.testing.assert_frame_equal(pd_es['new_entity'].df.reset_index(drop=True), ks_es['new_entity'].df.to_pandas())


@pytest.mark.skipif('not ks')
def test_create_entityset_with_mixed_dataframe_types(pd_es, ks_es):
    df = pd.DataFrame({"id": [0, 1, 2, 3],
                       "values": [1, 12, -34, 27]})
    ks_df = ks.from_pandas(df)

    # Test error is raised when trying to add Koalas entity to entitset with existing pandas entities
    err_msg = "All entity dataframes must be of the same type. " \
              "Cannot add entity of type {} to an entityset with existing entities " \
              "of type {}".format(type(ks_df), type(pd_es.entities[0].df))

    with pytest.raises(ValueError, match=err_msg):
        pd_es.entity_from_dataframe(
            entity_id="new_entity",
            dataframe=ks_df,
            index="id")

    # Test error is raised when trying to add pandas entity to entitset with existing ks entities
    err_msg = "All entity dataframes must be of the same type. " \
              "Cannot add entity of type {} to an entityset with existing entities " \
              "of type {}".format(type(df), type(ks_es.entities[0].df))

    with pytest.raises(ValueError, match=err_msg):
        ks_es.entity_from_dataframe(
            entity_id="new_entity",
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
    sessions_vtypes = {
        "id": ft.variable_types.Id,
        "user": ft.variable_types.Id,
        "time": ft.variable_types.DatetimeTimeIndex,
        "strings": ft.variable_types.NaturalLanguage
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
    transactions_vtypes = {
        "id": ft.variable_types.Id,
        "session_id": ft.variable_types.Id,
        "amount": ft.variable_types.Numeric,
        "time": ft.variable_types.DatetimeTimeIndex,
    }

    pd_es.entity_from_dataframe(entity_id="sessions", dataframe=sessions, index="id", time_index="time")
    ks_es.entity_from_dataframe(entity_id="sessions", dataframe=sessions_ks, index="id", time_index="time", variable_types=sessions_vtypes)

    pd_es.entity_from_dataframe(entity_id="transactions", dataframe=transactions, index="id", time_index="time")
    ks_es.entity_from_dataframe(entity_id="transactions", dataframe=transactions_ks, index="id", time_index="time", variable_types=transactions_vtypes)

    new_rel = Relationship(pd_es["sessions"]["id"], pd_es["transactions"]["session_id"])
    ks_rel = Relationship(ks_es["sessions"]["id"], ks_es["transactions"]["session_id"])

    pd_es = pd_es.add_relationship(new_rel)
    ks_es = ks_es.add_relationship(ks_rel)

    assert pd_es['sessions'].last_time_index is None
    assert ks_es['sessions'].last_time_index is None

    pd_es.add_last_time_indexes()
    ks_es.add_last_time_indexes()

    pd.testing.assert_series_equal(pd_es['sessions'].last_time_index.sort_index(), ks_es['sessions'].last_time_index.to_pandas().sort_index(), check_names=False)


@pytest.mark.skipif('not ks')
def test_create_entity_with_make_index():
    values = [1, 12, -23, 27]
    df = pd.DataFrame({"values": values})
    ks_df = ks.from_pandas(df)
    ks_es = EntitySet(id="ks_es")
    vtypes = {"values": ft.variable_types.Numeric}
    ks_es.entity_from_dataframe(entity_id="new_entity", dataframe=ks_df, make_index=True, index="new_index", variable_types=vtypes)

    expected_df = pd.DataFrame({"new_index": range(len(values)), "values": values})
    pd.testing.assert_frame_equal(expected_df, ks_es['new_entity'].df.to_pandas().sort_index())


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
    vtypes = {
        "id": ft.variable_types.Id,
        "values": ft.variable_types.Numeric,
        "dates": ft.variable_types.Datetime,
        "strings": ft.variable_types.NaturalLanguage
    }
    ks_es.entity_from_dataframe(entity_id="data",
                                dataframe=values_dd,
                                index="id",
                                variable_types=vtypes)

    ks_fm, _ = ft.dfs(entityset=ks_es,
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
    vtypes = {
        "id": ft.variable_types.Id,
        "values": ft.variable_types.Numeric,
        "dates": ft.variable_types.Datetime,
        "strings": ft.variable_types.NaturalLanguage
    }
    ks_es.entity_from_dataframe(entity_id="data",
                                dataframe=values_dd,
                                index="id",
                                variable_types=vtypes)

    ks_fm, _ = ft.dfs(entityset=ks_es,
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
    vtypes = {
        "id": ft.variable_types.Id,
        "values": ft.variable_types.Numeric,
        "dates": ft.variable_types.Datetime,
        "strings": ft.variable_types.NaturalLanguage
    }
    ks_es.entity_from_dataframe(entity_id="data",
                                dataframe=values_dd,
                                index="id",
                                variable_types=vtypes)

    ks_fm, _ = ft.dfs(entityset=ks_es,
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
    vtypes = {
        "id": ft.variable_types.Id,
        "values": ft.variable_types.Numeric,
        "dates": ft.variable_types.Datetime,
        "strings": ft.variable_types.NaturalLanguage
    }
    ks_es.entity_from_dataframe(entity_id="data",
                                dataframe=values_dd,
                                index="id",
                                variable_types=vtypes)

    ks_fm, _ = ft.dfs(entityset=ks_es,
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
    vtypes = {
        "id": ft.variable_types.Id,
        "values": ft.variable_types.Numeric,
        "dates": ft.variable_types.Datetime,
        "strings": ft.variable_types.NaturalLanguage
    }
    ks_es.entity_from_dataframe(entity_id="data",
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

    ks_fm, _ = ft.dfs(entityset=ks_es,
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
    vtypes = {
        "id": ft.variable_types.Id,
        "values": ft.variable_types.Numeric,
        "dates": ft.variable_types.Datetime,
    }
    ks_es.entity_from_dataframe(entity_id="data",
                                dataframe=values_dd,
                                index="id",
                                time_index="dates",
                                variable_types=vtypes)

    ks_fm, _ = ft.dfs(entityset=ks_es,
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

    pd.testing.assert_frame_equal(fm, ks_fm.to_pandas().set_index('id').loc[fm.index])


@pytest.mark.skipif('not ks')
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
    log_ks = ks.from_pandas(log_df)

    flights_df = pd.DataFrame()
    flights_df['id'] = [0, 1, 2, 3]
    flights_df['origin'] = ["BOS", "LAX", "BOS", "LAX"]
    flights_ks = ks.from_pandas(flights_df)

    pd_es = ft.EntitySet("flights")
    ks_es = ft.EntitySet("flights_ks")

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
    ks_es.entity_from_dataframe(entity_id='logs',
                                dataframe=log_ks,
                                index="id",
                                variable_types=log_vtypes,
                                time_index="scheduled_time",
                                secondary_time_index={
                                      'arrival_time': ['departure_time', 'delay']})

    pd_es.entity_from_dataframe('flights', flights_df, index="id")
    flights_vtypes = pd_es['flights'].variable_types
    ks_es.entity_from_dataframe('flights', flights_ks, index="id", variable_types=flights_vtypes)

    new_rel = ft.Relationship(pd_es['flights']['id'], pd_es['logs']['flight_id'])
    ks_rel = ft.Relationship(ks_es['flights']['id'], ks_es['logs']['flight_id'])
    pd_es.add_relationship(new_rel)
    ks_es.add_relationship(ks_rel)

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
