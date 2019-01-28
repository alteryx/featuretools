# -*- coding: utf-8 -*-

import copy
import os
import shutil
import tempfile
from builtins import range
from datetime import datetime
from itertools import combinations
from random import randint

import numpy as np
import pandas as pd
import psutil
import pytest
from distributed.utils_test import cluster

from ..testing_utils import MockClient, make_ecommerce_entityset, mock_cluster

import featuretools as ft
from featuretools import EntitySet, Timedelta, calculate_feature_matrix, dfs
from featuretools.computational_backends.utils import (
    bin_cutoff_times,
    calc_num_per_chunk,
    create_client_and_cluster,
    get_next_chunk,
    n_jobs_to_workers
)
from featuretools.feature_base import (
    AggregationFeature,
    DirectFeature,
    IdentityFeature
)
from featuretools.primitives import Count, Max, Min, Percentile, Sum


@pytest.fixture(scope='module')
def entityset():
    return make_ecommerce_entityset()


@pytest.fixture
def int_es():
    return make_ecommerce_entityset(with_integer_time_index=True)


# TODO test mean ignores nan values
def test_calc_feature_matrix(entityset):
    times = list([datetime(2011, 4, 9, 10, 30, i * 6) for i in range(5)] +
                 [datetime(2011, 4, 9, 10, 31, i * 9) for i in range(4)] +
                 [datetime(2011, 4, 9, 10, 40, 0)] +
                 [datetime(2011, 4, 10, 10, 40, i) for i in range(2)] +
                 [datetime(2011, 4, 10, 10, 41, i * 3) for i in range(3)] +
                 [datetime(2011, 4, 10, 11, 10, i * 3) for i in range(2)])
    instances = range(17)
    cutoff_time = pd.DataFrame({'time': times, entityset['log'].index: instances})
    labels = [False] * 3 + [True] * 2 + [False] * 9 + [True] + [False] * 2

    property_feature = ft.Feature(entityset['log']['value']) > 10

    feature_matrix = calculate_feature_matrix([property_feature],
                                              entityset,
                                              cutoff_time=cutoff_time,
                                              verbose=True)

    assert (feature_matrix[property_feature.get_name()] == labels).values.all()

    error_text = 'features must be a non-empty list of features'
    with pytest.raises(AssertionError, match=error_text):
        feature_matrix = calculate_feature_matrix('features', entityset, cutoff_time=cutoff_time)

    with pytest.raises(AssertionError, match=error_text):
        feature_matrix = calculate_feature_matrix([], entityset, cutoff_time=cutoff_time)

    with pytest.raises(AssertionError, match=error_text):
        feature_matrix = calculate_feature_matrix([1, 2, 3], entityset, cutoff_time=cutoff_time)

    error_text = "cutoff_time times must be datetime type: try casting via "\
        "pd\\.to_datetime\\(cutoff_time\\['time'\\]\\)"
    with pytest.raises(TypeError, match=error_text):
        calculate_feature_matrix([property_feature],
                                 entityset,
                                 instance_ids=range(17),
                                 cutoff_time=17)

    error_text = 'cutoff_time must be a single value or DataFrame'
    with pytest.raises(TypeError, match=error_text):
        calculate_feature_matrix([property_feature],
                                 entityset,
                                 instance_ids=range(17),
                                 cutoff_time=times)

    cutoff_times_dup = pd.DataFrame({'time': [pd.datetime(2018, 3, 1),
                                              pd.datetime(2018, 3, 1)],
                                     entityset['log'].index: [1, 1]})

    error_text = 'Duplicated rows in cutoff time dataframe.'
    with pytest.raises(AssertionError, match=error_text):
        feature_matrix = calculate_feature_matrix([property_feature],
                                                  entityset=entityset,
                                                  cutoff_time=cutoff_times_dup)


def test_cfm_approximate_correct_ordering():
    trips = {
        'trip_id': [i for i in range(1000)],
        'flight_time': [datetime(1998, 4, 2) for i in range(350)] + [datetime(1997, 4, 3) for i in range(650)],
        'flight_id': [randint(1, 25) for i in range(1000)],
        'trip_duration': [randint(1, 999) for i in range(1000)]
    }
    df = pd.DataFrame.from_dict(trips)
    es = EntitySet('flights')
    es.entity_from_dataframe("trips",
                             dataframe=df,
                             index="trip_id",
                             time_index='flight_time')
    es.normalize_entity(base_entity_id="trips",
                        new_entity_id="flights",
                        index="flight_id",
                        make_time_index=True)
    features = dfs(entityset=es, target_entity='trips', features_only=True)
    flight_features = [feature for feature in features
                       if isinstance(feature, DirectFeature) and
                       isinstance(feature.base_features[0],
                                  AggregationFeature)]
    property_feature = IdentityFeature(es['trips']['trip_id'])
    # direct_agg_feat = DirectFeature(Sum(es['trips']['trip_duration'],
    #                                     es['flights']),
    #                                 es['trips'])
    cutoff_time = pd.DataFrame.from_dict({'instance_id': df['trip_id'],
                                          'time': df['flight_time']})
    time_feature = IdentityFeature(es['trips']['flight_time'])
    feature_matrix = calculate_feature_matrix(flight_features + [property_feature, time_feature],
                                              es,
                                              cutoff_time_in_index=True,
                                              cutoff_time=cutoff_time)
    feature_matrix.index.names = ['instance', 'time']
    assert(np.all(feature_matrix.reset_index('time').reset_index()[['instance', 'time']].values == feature_matrix[['trip_id', 'flight_time']].values))
    feature_matrix_2 = calculate_feature_matrix(flight_features + [property_feature, time_feature],
                                                es,
                                                cutoff_time=cutoff_time,
                                                cutoff_time_in_index=True,
                                                approximate=Timedelta(2, 'd'))
    feature_matrix_2.index.names = ['instance', 'time']
    assert(np.all(feature_matrix_2.reset_index('time').reset_index()[['instance', 'time']].values == feature_matrix_2[['trip_id', 'flight_time']].values))
    for column in feature_matrix:
        for x, y in zip(feature_matrix[column], feature_matrix_2[column]):
            assert ((pd.isnull(x) and pd.isnull(y)) or (x == y))


def test_cfm_no_cutoff_time_index(entityset):
    es = entityset
    agg_feat = ft.Feature(es['log']['id'], parent_entity=es['sessions'], primitive=Count)
    agg_feat4 = ft.Feature(agg_feat, parent_entity=es['customers'], primitive=Sum)
    dfeat = DirectFeature(agg_feat4, es['sessions'])
    cutoff_time = pd.DataFrame({
        'time': [datetime(2013, 4, 9, 10, 31, 19), datetime(2013, 4, 9, 11, 0, 0)],
        'instance_id': [0, 2]
    })
    feature_matrix = calculate_feature_matrix([dfeat, agg_feat],
                                              entityset,
                                              cutoff_time_in_index=False,
                                              approximate=Timedelta(12, 's'),
                                              cutoff_time=cutoff_time)
    assert feature_matrix.index.name == 'id'
    assert feature_matrix.index.values.tolist() == [0, 2]
    assert feature_matrix[dfeat.get_name()].tolist() == [10, 10]
    assert feature_matrix[agg_feat.get_name()].tolist() == [5, 1]

    cutoff_time = pd.DataFrame({
        'time': [datetime(2011, 4, 9, 10, 31, 19), datetime(2011, 4, 9, 11, 0, 0)],
        'instance_id': [0, 2]
    })
    feature_matrix_2 = calculate_feature_matrix([dfeat, agg_feat],
                                                entityset,
                                                cutoff_time_in_index=False,
                                                approximate=Timedelta(10, 's'),
                                                cutoff_time=cutoff_time)
    assert feature_matrix_2.index.name == 'id'
    assert feature_matrix_2.index.tolist() == [0, 2]
    assert feature_matrix_2[dfeat.get_name()].tolist() == [7, 10]
    assert feature_matrix_2[agg_feat.get_name()].tolist() == [5, 1]


def test_cfm_duplicated_index_in_cutoff_time(entityset):
    times = [pd.datetime(2011, 4, 1), pd.datetime(2011, 5, 1),
             pd.datetime(2011, 4, 1), pd.datetime(2011, 5, 1)]

    instances = [1, 1, 2, 2]
    property_feature = ft.Feature(entityset['log']['value']) > 10
    cutoff_time = pd.DataFrame({'id': instances, 'time': times},
                               index=[1, 1, 1, 1])

    feature_matrix = calculate_feature_matrix([property_feature],
                                              entityset,
                                              cutoff_time=cutoff_time,
                                              chunk_size=1)

    assert (feature_matrix.shape[0] == cutoff_time.shape[0])


def test_saveprogress(entityset):
    times = list([datetime(2011, 4, 9, 10, 30, i * 6) for i in range(5)] +
                 [datetime(2011, 4, 9, 10, 31, i * 9) for i in range(4)] +
                 [datetime(2011, 4, 9, 10, 40, 0)] +
                 [datetime(2011, 4, 10, 10, 40, i) for i in range(2)] +
                 [datetime(2011, 4, 10, 10, 41, i * 3) for i in range(3)] +
                 [datetime(2011, 4, 10, 11, 10, i * 3) for i in range(2)])
    cutoff_time = pd.DataFrame({'time': times, 'instance_id': range(17)})
    property_feature = ft.Feature(entityset['log']['value']) > 10
    save_progress = tempfile.mkdtemp()
    fm_save = calculate_feature_matrix([property_feature],
                                       entityset,
                                       cutoff_time=cutoff_time,
                                       save_progress=save_progress)
    _, _, files = next(os.walk(save_progress))
    files = [os.path.join(save_progress, file) for file in files]
    # there is 17 datetime files created above
    assert len(files) == 17
    list_df = []
    for file_ in files:
        df = pd.read_csv(file_, index_col="id", header=0)
        list_df.append(df)
    merged_df = pd.concat(list_df)
    merged_df.set_index(pd.DatetimeIndex(times), inplace=True, append=True)
    fm_no_save = calculate_feature_matrix([property_feature],
                                          entityset,
                                          cutoff_time=cutoff_time)
    assert np.all((merged_df.sort_index().values) == (fm_save.sort_index().values))
    assert np.all((fm_no_save.sort_index().values) == (fm_save.sort_index().values))
    assert np.all((fm_no_save.sort_index().values) == (merged_df.sort_index().values))
    shutil.rmtree(save_progress)


def test_cutoff_time_correctly(entityset):
    property_feature = ft.Feature(entityset['log']['id'], parent_entity=entityset['customers'], primitive=Count)
    times = [datetime(2011, 4, 10), datetime(2011, 4, 11), datetime(2011, 4, 7)]
    cutoff_time = pd.DataFrame({'time': times, 'instance_id': [0, 1, 2]})
    feature_matrix = calculate_feature_matrix([property_feature],
                                              entityset,
                                              cutoff_time=cutoff_time)

    labels = [0, 10, 5]
    assert (feature_matrix[property_feature.get_name()] == labels).values.all()


def test_cutoff_time_binning(entityset):
    cutoff_time = pd.DataFrame({'time': [
                                datetime(2011, 4, 9, 12, 31),
                                datetime(2011, 4, 10, 11),
                                datetime(2011, 4, 10, 13, 10, 1)
                                ],
                                'instance_id': [1, 2, 3]})
    binned_cutoff_times = bin_cutoff_times(cutoff_time, Timedelta(4, 'h'))
    labels = [datetime(2011, 4, 9, 12),
              datetime(2011, 4, 10, 8),
              datetime(2011, 4, 10, 12)]
    for i in binned_cutoff_times.index:
        assert binned_cutoff_times['time'][i] == labels[i]

    binned_cutoff_times = bin_cutoff_times(cutoff_time, Timedelta(25, 'h'))
    labels = [datetime(2011, 4, 8, 22),
              datetime(2011, 4, 9, 23),
              datetime(2011, 4, 9, 23)]
    for i in binned_cutoff_times.index:
        assert binned_cutoff_times['time'][i] == labels[i]


def test_training_window(entityset):
    property_feature = ft.Feature(entityset['log']['id'], parent_entity=entityset['customers'], primitive=Count)
    top_level_agg = ft.Feature(entityset['customers']['id'], parent_entity=entityset[u'régions'], primitive=Count)

    # make sure features that have a direct to a higher level agg
    # so we have multiple "filter eids" in get_pandas_data_slice,
    # and we go through the loop to pull data with a training_window param more than once
    dagg = DirectFeature(top_level_agg, entityset['customers'])

    # for now, warns if last_time_index not present
    times = [datetime(2011, 4, 9, 12, 31),
             datetime(2011, 4, 10, 11),
             datetime(2011, 4, 10, 13, 10, 1)]
    cutoff_time = pd.DataFrame({'time': times, 'instance_id': [0, 1, 2]})
    feature_matrix = calculate_feature_matrix([property_feature, dagg],
                                              entityset,
                                              cutoff_time=cutoff_time,
                                              training_window='2 hours')

    entityset.add_last_time_indexes()

    error_text = 'training window must be an absolute Timedelta'
    with pytest.raises(AssertionError, match=error_text):
        feature_matrix = calculate_feature_matrix([property_feature],
                                                  entityset,
                                                  cutoff_time=cutoff_time,
                                                  training_window=Timedelta(2, 'observations', entity='log'))

    feature_matrix = calculate_feature_matrix([property_feature, dagg],
                                              entityset,
                                              cutoff_time=cutoff_time,
                                              training_window='2 hours')
    prop_values = [5, 5, 1]
    dagg_values = [3, 2, 1]
    assert (feature_matrix[property_feature.get_name()] == prop_values).values.all()
    assert (feature_matrix[dagg.get_name()] == dagg_values).values.all()


def test_training_window_recent_time_index(entityset):
    # customer with no sessions
    row = {
        'id': [3],
        'age': [73],
        u'région_id': ['United States'],
        'cohort': [1],
        'cancel_reason': ["I am finally awake!!"],
        'loves_ice_cream': [True],
        'favorite_quote': ["Who is John Galt?"],
        'signup_date': [datetime(2011, 4, 10)],
        'upgrade_date': [datetime(2011, 4, 12)],
        'cancel_date': [datetime(2011, 5, 13)],
        'date_of_birth': [datetime(1938, 2, 1)],
        'engagement_level': [2],
    }
    to_add_df = pd.DataFrame(row)
    to_add_df.index = range(3, 4)

    # have to convert category to int in order to concat
    old_df = entityset['customers'].df
    old_df.index = old_df.index.astype("int")
    old_df["id"] = old_df["id"].astype(int)

    df = pd.concat([old_df, to_add_df], sort=True)

    # convert back after
    df.index = df.index.astype("category")
    df["id"] = df["id"].astype("category")

    entityset['customers'].update_data(df=df,
                                       recalculate_last_time_indexes=False)
    entityset.add_last_time_indexes()

    property_feature = ft.Feature(entityset['log']['id'], parent_entity=entityset['customers'], primitive=Count)
    top_level_agg = ft.Feature(entityset['customers']['id'], parent_entity=entityset[u'régions'], primitive=Count)
    dagg = DirectFeature(top_level_agg, entityset['customers'])
    instance_ids = [0, 1, 2, 3]
    times = [datetime(2011, 4, 9, 12, 31), datetime(2011, 4, 10, 11),
             datetime(2011, 4, 10, 13, 10, 1), datetime(2011, 4, 10, 1, 59, 59)]
    cutoff_time = pd.DataFrame({'time': times, 'instance_id': instance_ids})
    feature_matrix = calculate_feature_matrix(
        [property_feature, dagg],
        entityset,
        cutoff_time=cutoff_time,
        training_window='2 hours'
    )
    prop_values = [5, 5, 1, 0]
    dagg_values = [3, 2, 1, 3]
    feature_matrix.sort_index(inplace=True)
    assert (feature_matrix[property_feature.get_name()] == prop_values).values.all()
    assert (feature_matrix[dagg.get_name()] == dagg_values).values.all()


def test_approximate_multiple_instances_per_cutoff_time(entityset):
    es = entityset
    agg_feat = ft.Feature(es['log']['id'], parent_entity=es['sessions'], primitive=Count)
    agg_feat2 = ft.Feature(agg_feat, parent_entity=es['customers'], primitive=Sum)
    dfeat = DirectFeature(agg_feat2, es['sessions'])
    times = [datetime(2011, 4, 9, 10, 31, 19), datetime(2011, 4, 9, 11, 0, 0)]
    cutoff_time = pd.DataFrame({'time': times, 'instance_id': [0, 2]})
    feature_matrix = calculate_feature_matrix([dfeat, agg_feat],
                                              entityset,
                                              approximate=Timedelta(1, 'week'),
                                              cutoff_time=cutoff_time,
                                              chunk_size="cutoff time")
    assert feature_matrix.shape[0] == 2
    assert feature_matrix[agg_feat.get_name()].tolist() == [5, 1]


def test_approximate_dfeat_of_agg_on_target(entityset):
    es = entityset
    agg_feat = ft.Feature(es['log']['id'], parent_entity=es['sessions'], primitive=Count)
    agg_feat2 = ft.Feature(agg_feat, parent_entity=es['customers'], primitive=Sum)
    dfeat = DirectFeature(agg_feat2, es['sessions'])
    times = [datetime(2011, 4, 9, 10, 31, 19), datetime(2011, 4, 9, 11, 0, 0)]
    cutoff_time = pd.DataFrame({'time': times, 'instance_id': [0, 2]})
    feature_matrix = calculate_feature_matrix([dfeat, agg_feat],
                                              entityset,
                                              instance_ids=[0, 2],
                                              approximate=Timedelta(10, 's'),
                                              cutoff_time=cutoff_time)
    assert feature_matrix[dfeat.get_name()].tolist() == [7, 10]
    assert feature_matrix[agg_feat.get_name()].tolist() == [5, 1]


def test_approximate_dfeat_of_need_all_values(entityset):
    es = entityset
    p = ft.Feature(es['log']['value'], primitive=Percentile)
    agg_feat = ft.Feature(p, parent_entity=es['sessions'], primitive=Sum)
    agg_feat2 = ft.Feature(agg_feat, parent_entity=es['customers'], primitive=Sum)
    dfeat = DirectFeature(agg_feat2, es['sessions'])
    times = [datetime(2011, 4, 9, 10, 31, 19), datetime(2011, 4, 9, 11, 0, 0)]
    cutoff_time = pd.DataFrame({'time': times, 'instance_id': [0, 2]})
    feature_matrix = calculate_feature_matrix([dfeat, agg_feat],
                                              entityset,
                                              approximate=Timedelta(10, 's'),
                                              cutoff_time_in_index=True,
                                              cutoff_time=cutoff_time)
    log_df = es['log'].df
    instances = [0, 2]
    cutoffs = [pd.Timestamp('2011-04-09 10:31:19'), pd.Timestamp('2011-04-09 11:00:00')]
    approxes = [pd.Timestamp('2011-04-09 10:31:10'), pd.Timestamp('2011-04-09 11:00:00')]
    true_vals = []
    true_vals_approx = []
    for instance, cutoff, approx in zip(instances, cutoffs, approxes):
        log_data_cutoff = log_df[log_df['datetime'] < cutoff]
        log_data_cutoff['percentile'] = log_data_cutoff['value'].rank(pct=True)
        true_agg = log_data_cutoff.loc[log_data_cutoff['session_id'] == instance, 'percentile'].fillna(0).sum()
        true_vals.append(round(true_agg, 3))

        log_data_approx = log_df[log_df['datetime'] < approx]
        log_data_approx['percentile'] = log_data_approx['value'].rank(pct=True)
        true_agg_approx = log_data_approx.loc[log_data_approx['session_id'].isin([0, 1, 2]), 'percentile'].fillna(0).sum()
        true_vals_approx.append(round(true_agg_approx, 3))
    lapprox = [round(x, 3) for x in feature_matrix[dfeat.get_name()].tolist()]
    test_list = [round(x, 3) for x in feature_matrix[agg_feat.get_name()].tolist()]
    assert lapprox == true_vals_approx
    assert test_list == true_vals


def test_uses_full_entity_feat_of_approximate(entityset):
    es = entityset
    agg_feat = ft.Feature(es['log']['value'], parent_entity=es['sessions'], primitive=Sum)
    agg_feat2 = ft.Feature(agg_feat, parent_entity=es['customers'], primitive=Sum)
    agg_feat3 = ft.Feature(agg_feat, parent_entity=es['customers'], primitive=Max)
    dfeat = DirectFeature(agg_feat2, es['sessions'])
    dfeat2 = DirectFeature(agg_feat3, es['sessions'])
    p = ft.Feature(dfeat, primitive=Percentile)
    times = [datetime(2011, 4, 9, 10, 31, 19), datetime(2011, 4, 9, 11, 0, 0)]
    cutoff_time = pd.DataFrame({'time': times, 'instance_id': [0, 2]})
    # only dfeat2 should be approximated
    # because Percentile needs all values

    feature_matrix_only_dfeat2 = calculate_feature_matrix(
        [dfeat2],
        entityset,
        approximate=Timedelta(10, 's'),
        cutoff_time_in_index=True,
        cutoff_time=cutoff_time)
    assert feature_matrix_only_dfeat2[dfeat2.get_name()].tolist() == [50, 50]

    feature_matrix_approx = calculate_feature_matrix(
        [p, dfeat, dfeat2, agg_feat],
        entityset,
        approximate=Timedelta(10, 's'),
        cutoff_time_in_index=True,
        cutoff_time=cutoff_time)
    assert feature_matrix_only_dfeat2[dfeat2.get_name()].tolist() == feature_matrix_approx[dfeat2.get_name()].tolist()

    feature_matrix_small_approx = calculate_feature_matrix(
        [p, dfeat, dfeat2, agg_feat],
        entityset,
        approximate=Timedelta(10, 'ms'),
        cutoff_time_in_index=True,
        cutoff_time=cutoff_time)

    feature_matrix_no_approx = calculate_feature_matrix(
        [p, dfeat, dfeat2, agg_feat],
        entityset,
        cutoff_time_in_index=True,
        cutoff_time=cutoff_time)
    for f in [p, dfeat, agg_feat]:
        for fm1, fm2 in combinations([feature_matrix_approx,
                                      feature_matrix_small_approx,
                                      feature_matrix_no_approx], 2):
            assert fm1[f.get_name()].tolist() == fm2[f.get_name()].tolist()


def test_approximate_dfeat_of_dfeat_of_agg_on_target(entityset):
    es = entityset
    agg_feat = ft.Feature(es['log']['id'], parent_entity=es['sessions'], primitive=Count)
    agg_feat2 = ft.Feature(agg_feat, parent_entity=es['customers'], primitive=Sum)
    dfeat = DirectFeature(ft.Feature(agg_feat2, es["sessions"]), es['log'])
    times = [datetime(2011, 4, 9, 10, 31, 19), datetime(2011, 4, 9, 11, 0, 0)]
    cutoff_time = pd.DataFrame({'time': times, 'instance_id': [0, 2]})
    feature_matrix = calculate_feature_matrix([dfeat],
                                              entityset,
                                              approximate=Timedelta(10, 's'),
                                              cutoff_time=cutoff_time)
    assert feature_matrix[dfeat.get_name()].tolist() == [7, 10]


def test_empty_path_approximate_full(entityset):
    es = copy.deepcopy(entityset)
    es['sessions'].df['customer_id'] = [np.nan, np.nan, np.nan, 1, 1, 2]
    agg_feat = ft.Feature(es['log']['id'], parent_entity=es['sessions'], primitive=Count)
    agg_feat2 = ft.Feature(agg_feat, parent_entity=es['customers'], primitive=Sum)
    dfeat = DirectFeature(agg_feat2, es['sessions'])
    times = [datetime(2011, 4, 9, 10, 31, 19), datetime(2011, 4, 9, 11, 0, 0)]
    cutoff_time = pd.DataFrame({'time': times, 'instance_id': [0, 2]})
    feature_matrix = calculate_feature_matrix([dfeat, agg_feat],
                                              es,
                                              approximate=Timedelta(10, 's'),
                                              cutoff_time=cutoff_time)
    vals1 = feature_matrix[dfeat.get_name()].tolist()
    assert np.isnan(vals1[0])
    assert np.isnan(vals1[1])
    assert feature_matrix[agg_feat.get_name()].tolist() == [5, 1]

# todo: do we need to test this situation?
# def test_empty_path_approximate_partial(entityset):
#     es = copy.deepcopy(entityset)
#     es['sessions'].df['customer_id'] = pd.Categorical([0, 0, np.nan, 1, 1, 2])
#     agg_feat = ft.Feature(es['log']['id'], parent_entity=es['sessions'], primitive=Count)
#     agg_feat2 = ft.Feature(agg_feat, parent_entity=es['customers'], primitive=Sum)
#     dfeat = DirectFeature(agg_feat2, es['sessions'])
#     times = [datetime(2011, 4, 9, 10, 31, 19), datetime(2011, 4, 9, 11, 0, 0)]
#     cutoff_time = pd.DataFrame({'time': times, 'instance_id': [0, 2]})
#     feature_matrix = calculate_feature_matrix([dfeat, agg_feat],
#                                               es,
#                                               approximate=Timedelta(10, 's'),
#                                               cutoff_time=cutoff_time)
#     vals1 = feature_matrix[dfeat.get_name()].tolist()
#     assert vals1[0] == 7
#     assert np.isnan(vals1[1])
#     assert feature_matrix[agg_feat.get_name()].tolist() == [5, 1]


def test_approx_base_feature_is_also_first_class_feature(entityset):
    es = entityset
    log_to_products = DirectFeature(es['products']['rating'], es['log'])
    # This should still be computed properly
    agg_feat = ft.Feature(log_to_products, parent_entity=es['sessions'], primitive=Min)
    customer_agg_feat = ft.Feature(agg_feat, parent_entity=es['customers'], primitive=Sum)
    # This is to be approximated
    sess_to_cust = DirectFeature(customer_agg_feat, es['sessions'])
    times = [datetime(2011, 4, 9, 10, 31, 19), datetime(2011, 4, 9, 11, 0, 0)]
    cutoff_time = pd.DataFrame({'time': times, 'instance_id': [0, 2]})
    feature_matrix = calculate_feature_matrix([sess_to_cust, agg_feat],
                                              entityset,
                                              approximate=Timedelta(10, 's'),
                                              cutoff_time=cutoff_time)
    vals1 = feature_matrix[sess_to_cust.get_name()].tolist()
    assert vals1 == [8.5, 7]
    vals2 = feature_matrix[agg_feat.get_name()].tolist()
    assert vals2 == [4, 1.5]


def test_approximate_time_split_returns_the_same_result(entityset):
    es = entityset
    agg_feat = ft.Feature(es['log']['id'], parent_entity=es['sessions'], primitive=Count)
    agg_feat2 = ft.Feature(agg_feat, parent_entity=es['customers'], primitive=Sum)
    dfeat = DirectFeature(agg_feat2, es['sessions'])

    cutoff_df = pd.DataFrame({'time': [pd.Timestamp('2011-04-09 10:07:30'),
                                       pd.Timestamp('2011-04-09 10:07:40')],
                              'instance_id': [0, 0]})

    feature_matrix_at_once = calculate_feature_matrix([dfeat, agg_feat],
                                                      entityset,
                                                      approximate=Timedelta(10, 's'),
                                                      cutoff_time=cutoff_df)
    divided_matrices = []
    separate_cutoff = [cutoff_df.iloc[0:1], cutoff_df.iloc[1:]]
    # Make sure indexes are different
    # Not that this step is unecessary and done to showcase the issue here
    separate_cutoff[0].index = [0]
    separate_cutoff[1].index = [1]
    for ct in separate_cutoff:
        fm = calculate_feature_matrix([dfeat, agg_feat],
                                      entityset,
                                      approximate=Timedelta(10, 's'),
                                      cutoff_time=ct)
        divided_matrices.append(fm)
    feature_matrix_from_split = pd.concat(divided_matrices)
    assert feature_matrix_from_split.shape == feature_matrix_at_once.shape
    for i1, i2 in zip(feature_matrix_at_once.index, feature_matrix_from_split.index):
        assert (pd.isnull(i1) and pd.isnull(i2)) or (i1 == i2)
    for c in feature_matrix_from_split:
        for i1, i2 in zip(feature_matrix_at_once[c], feature_matrix_from_split[c]):
            assert (pd.isnull(i1) and pd.isnull(i2)) or (i1 == i2)


def test_approximate_returns_correct_empty_default_values(entityset):
    es = entityset
    agg_feat = ft.Feature(es['log']['id'], parent_entity=es['customers'], primitive=Count)
    dfeat = DirectFeature(agg_feat, es['sessions'])

    cutoff_df = pd.DataFrame({'time': [pd.Timestamp('2011-04-08 11:00:00'),
                                       pd.Timestamp('2011-04-09 11:00:00')],
                              'instance_id': [0, 0]})

    fm = calculate_feature_matrix([dfeat],
                                  entityset,
                                  approximate=Timedelta(10, 's'),
                                  cutoff_time=cutoff_df)
    assert fm[dfeat.get_name()].tolist() == [0, 10]


# def test_approximate_deep_recurse(entityset):
    # es = entityset
    # agg_feat = ft.Feature(es['customers']['id'], parent_entity=es[u'régions'], primitive=Count)
    # dfeat1 = DirectFeature(agg_feat, es['sessions'])
    # agg_feat2 = Sum(dfeat1, es['customers'])
    # dfeat2 = DirectFeature(agg_feat2, es['sessions'])

    # agg_feat3 = ft.Feature(es['log']['id'], parent_entity=es['products'], primitive=Count)
    # dfeat3 = DirectFeature(agg_feat3, es['log'])
    # agg_feat4 = Sum(dfeat3, es['sessions'])

    # feature_matrix = calculate_feature_matrix([dfeat2, agg_feat4],
    #   entityset,
    #                                          instance_ids=[0, 2],
    #                                          approximate=Timedelta(10, 's'),
    #                                          cutoff_time=[datetime(2011, 4, 9, 10, 31, 19),
    #                                                       datetime(2011, 4, 9, 11, 0, 0)])
    # # dfeat2 and agg_feat4 should both be approximated


def test_approximate_child_aggs_handled_correctly(entityset):
    es = entityset
    agg_feat = ft.Feature(es['customers']['id'], parent_entity=es[u'régions'], primitive=Count)
    dfeat = DirectFeature(agg_feat, es['customers'])
    agg_feat_2 = ft.Feature(es['log']['value'], parent_entity=es['customers'], primitive=Sum)
    cutoff_df = pd.DataFrame({'time': [pd.Timestamp('2011-04-08 10:30:00'),
                                       pd.Timestamp('2011-04-09 10:30:06')],
                              'instance_id': [0, 0]})

    fm = calculate_feature_matrix([dfeat],
                                  entityset,
                                  approximate=Timedelta(10, 's'),
                                  cutoff_time=cutoff_df)
    fm_2 = calculate_feature_matrix([dfeat, agg_feat_2],
                                    entityset,
                                    approximate=Timedelta(10, 's'),
                                    cutoff_time=cutoff_df)
    assert fm[dfeat.get_name()].tolist() == [2, 3]
    assert fm_2[agg_feat_2.get_name()].tolist() == [0, 5]


def test_cutoff_time_naming(entityset):
    es = entityset

    agg_feat = ft.Feature(es['customers']['id'], parent_entity=es[u'régions'], primitive=Count)
    dfeat = DirectFeature(agg_feat, es['customers'])
    cutoff_df = pd.DataFrame({'time': [pd.Timestamp('2011-04-08 10:30:00'),
                                       pd.Timestamp('2011-04-09 10:30:06')],
                              'instance_id': [0, 0]})
    cutoff_df_index_name = cutoff_df.rename(columns={"instance_id": "id"})
    cutoff_df_time_name = cutoff_df.rename(columns={"time": "cutoff_time"})
    cutoff_df_index_name_time_name = cutoff_df.rename(columns={"instance_id": "id", "time": "cutoff_time"})
    cutoff_df_wrong_index_name = cutoff_df.rename(columns={"instance_id": "wrong_id"})

    fm1 = calculate_feature_matrix([dfeat], entityset, cutoff_time=cutoff_df)
    for test_cutoff in [cutoff_df_index_name, cutoff_df_time_name, cutoff_df_index_name_time_name]:
        fm2 = calculate_feature_matrix([dfeat], entityset, cutoff_time=test_cutoff)

        assert all((fm1 == fm2.values).values)

    error_text = 'Name of the index variable in the target entity or "instance_id" must be present in cutoff_time'
    with pytest.raises(AttributeError, match=error_text):
        calculate_feature_matrix([dfeat], entityset, cutoff_time=cutoff_df_wrong_index_name)


def test_cutoff_time_extra_columns(entityset):
    es = entityset

    agg_feat = ft.Feature(es['customers']['id'], parent_entity=es[u'régions'], primitive=Count)
    dfeat = DirectFeature(agg_feat, es['customers'])

    cutoff_df = pd.DataFrame({'time': [pd.Timestamp('2011-04-09 10:30:06'),
                                       pd.Timestamp('2011-04-09 10:30:03'),
                                       pd.Timestamp('2011-04-08 10:30:00')],
                              'instance_id': [0, 1, 0],
                              'label': [True, True, False]},
                             columns=['time', 'instance_id', 'label'])
    fm = calculate_feature_matrix([dfeat], entityset, cutoff_time=cutoff_df)
    # check column was added to end of matrix
    assert 'label' == fm.columns[-1]
    # check column was sorted by time labelike the rest of the feature matrix
    true_series = pd.Series([False, True, True], index=[0, 1, 0])
    assert (fm['label'] == true_series).all()

    fm_2 = calculate_feature_matrix([dfeat],
                                    entityset,
                                    cutoff_time=cutoff_df,
                                    approximate="2 days")
    # check column was added to end of matrix
    assert 'label' in fm_2.columns
    # check column was sorted by time like the rest of the feature matrix
    true_series = pd.Series([False, True, True], index=[0, 1, 0])
    assert (fm_2['label'] == true_series).all()


def test_cfm_returns_original_time_indexes(entityset):
    es = entityset

    agg_feat = ft.Feature(es['customers']['id'], parent_entity=es[u'régions'], primitive=Count)
    dfeat = DirectFeature(agg_feat, es['customers'])
    agg_feat_2 = ft.Feature(es['sessions']['id'], parent_entity=es['customers'], primitive=Count)
    cutoff_df = pd.DataFrame({'time': [pd.Timestamp('2011-04-09 10:30:06'),
                                       pd.Timestamp('2011-04-09 10:30:03'),
                                       pd.Timestamp('2011-04-08 10:30:00')],
                              'instance_id': [0, 1, 0]})
    sorted_df = cutoff_df.sort_values(['time', 'instance_id'], kind='mergesort')

    # no approximate
    fm = calculate_feature_matrix([dfeat],
                                  entityset, cutoff_time=cutoff_df,
                                  cutoff_time_in_index=True)
    instance_level_vals = fm.index.get_level_values(0).values
    time_level_vals = fm.index.get_level_values(1).values
    assert (instance_level_vals == sorted_df['instance_id'].values).all()
    assert (time_level_vals == sorted_df['time'].values).all()

    # approximate, in different windows, no unapproximated aggs
    fm2 = calculate_feature_matrix([dfeat], entityset, cutoff_time=cutoff_df,
                                   cutoff_time_in_index=True, approximate="1 m")
    instance_level_vals = fm2.index.get_level_values(0).values
    time_level_vals = fm2.index.get_level_values(1).values
    assert (instance_level_vals == sorted_df['instance_id'].values).all()
    assert (time_level_vals == sorted_df['time'].values).all()

    # approximate, in different windows, unapproximated aggs
    fm2 = calculate_feature_matrix([dfeat, agg_feat_2], entityset, cutoff_time=cutoff_df,
                                   cutoff_time_in_index=True, approximate="1 m")
    instance_level_vals = fm2.index.get_level_values(0).values
    time_level_vals = fm2.index.get_level_values(1).values
    assert (instance_level_vals == sorted_df['instance_id'].values).all()
    assert (time_level_vals == sorted_df['time'].values).all()

    # approximate, in same window, no unapproximated aggs
    fm3 = calculate_feature_matrix([dfeat], entityset, cutoff_time=cutoff_df,
                                   cutoff_time_in_index=True, approximate="2 d")
    instance_level_vals = fm3.index.get_level_values(0).values
    time_level_vals = fm3.index.get_level_values(1).values
    assert (instance_level_vals == sorted_df['instance_id'].values).all()
    assert (time_level_vals == sorted_df['time'].values).all()

    # approximate, in same window, unapproximated aggs
    fm3 = calculate_feature_matrix([dfeat, agg_feat_2], entityset, cutoff_time=cutoff_df,
                                   cutoff_time_in_index=True, approximate="2 d")
    instance_level_vals = fm3.index.get_level_values(0).values
    time_level_vals = fm3.index.get_level_values(1).values
    assert (instance_level_vals == sorted_df['instance_id'].values).all()
    assert (time_level_vals == sorted_df['time'].values).all()


def test_calculating_number_per_chunk():
    cutoff_df = pd.DataFrame({'time': [pd.Timestamp('2011-04-08 10:30:00')
                                       for x in range(200)],
                              'instance_id': [0 for x in range(200)]})

    singleton = pd.DataFrame({'time': [pd.Timestamp('2011-04-08 10:30:00')],
                              'instance_id': [0]})
    shape = cutoff_df.shape

    error_text = "chunk_size must be None, a float between 0 and 1,a positive integer, or the string 'cutoff time'"
    with pytest.raises(ValueError, match=error_text):
        calc_num_per_chunk(-1, shape)

    with pytest.raises(ValueError, match=error_text):
        calc_num_per_chunk("test", shape)

    with pytest.raises(ValueError, match=error_text):
        calc_num_per_chunk(2.5, shape)

    with pytest.warns(UserWarning):
        assert calc_num_per_chunk(201, shape) == 200

    assert calc_num_per_chunk(200, shape) == 200
    assert calc_num_per_chunk(11, shape) == 11
    assert calc_num_per_chunk(.7, shape) == 140
    assert calc_num_per_chunk(.6749, shape) == 134
    assert calc_num_per_chunk(.6751, shape) == 135
    assert calc_num_per_chunk(None, shape) == 20
    assert calc_num_per_chunk("cutoff time", shape) == "cutoff time"
    assert calc_num_per_chunk(1, shape) == 1
    assert calc_num_per_chunk(.5, singleton.shape) == 1
    assert calc_num_per_chunk(None, singleton.shape) == 10


def test_get_next_chunk(entityset):
    times = list([datetime(2011, 4, 9, 10, 30, i * 6) for i in range(5)] +
                 [datetime(2011, 4, 9, 10, 31, i * 9) for i in range(4)] +
                 [datetime(2011, 4, 9, 10, 40, 0)] +
                 [datetime(2011, 4, 10, 10, 40, i) for i in range(2)] +
                 [datetime(2011, 4, 10, 10, 41, i * 3) for i in range(3)] +
                 [datetime(2011, 4, 10, 11, 10, i * 3) for i in range(2)])
    cutoff_time = pd.DataFrame({'time': times, 'instance_id': range(17)})
    chunks = [chunk for chunk in get_next_chunk(cutoff_time, 'time', 4)]
    assert len(chunks) == 5

    # test when a cutoff time is larger than a chunk
    times = list([datetime(2011, 4, 9, 10, 30, 6) for i in range(5)] +
                 [datetime(2011, 4, 9, 10, 31, 9) for i in range(4)] +
                 [datetime(2011, 4, 9, 10, 40, 0)] +
                 [datetime(2011, 4, 10, 10, 40, i) for i in range(2)] +
                 [datetime(2011, 4, 10, 10, 41, i * 3) for i in range(3)] +
                 [datetime(2011, 4, 10, 11, 10, i * 3) for i in range(2)])
    cutoff_time = pd.DataFrame({'time': times, 'instance_id': range(17)})
    chunks = [chunk for chunk in get_next_chunk(cutoff_time, 'time', 4)]
    assert len(chunks) == 5
    # largest cutoff time handled first
    largest = pd.Series([datetime(2011, 4, 9, 10, 30, 6) for i in range(4)])
    assert (chunks[0]['time'] == largest).all()
    # additional part of cutoff time added to another chunk
    assert (chunks[2]['time'] == times[4]).any()

    # test when cutoff_time is smaller than num_per_chunk
    chunks = [chunk for chunk in get_next_chunk(cutoff_time, 'time', 18)]
    assert len(chunks) == 1


def test_verbose_cutoff_time_chunks(entityset):
    times = list([datetime(2011, 4, 9, 10, 30, i * 6) for i in range(5)] +
                 [datetime(2011, 4, 9, 10, 31, i * 9) for i in range(4)] +
                 [datetime(2011, 4, 9, 10, 40, 0)] +
                 [datetime(2011, 4, 10, 10, 40, i) for i in range(2)] +
                 [datetime(2011, 4, 10, 10, 41, i * 3) for i in range(3)] +
                 [datetime(2011, 4, 10, 11, 10, i * 3) for i in range(2)])
    labels = [False] * 3 + [True] * 2 + [False] * 9 + [True] + [False] * 2
    cutoff_time = pd.DataFrame({'time': times, 'instance_id': range(17)})
    property_feature = IdentityFeature(entityset['log']['value']) > 10

    feature_matrix = calculate_feature_matrix([property_feature],
                                              entityset,
                                              cutoff_time=cutoff_time,
                                              chunk_size="cutoff time",
                                              verbose=True)

    assert (feature_matrix[property_feature.get_name()] == labels).values.all()


def test_dask_kwargs(entityset):
    times = list([datetime(2011, 4, 9, 10, 30, i * 6) for i in range(5)] +
                 [datetime(2011, 4, 9, 10, 31, i * 9) for i in range(4)] +
                 [datetime(2011, 4, 9, 10, 40, 0)] +
                 [datetime(2011, 4, 10, 10, 40, i) for i in range(2)] +
                 [datetime(2011, 4, 10, 10, 41, i * 3) for i in range(3)] +
                 [datetime(2011, 4, 10, 11, 10, i * 3) for i in range(2)])
    labels = [False] * 3 + [True] * 2 + [False] * 9 + [True] + [False] * 2
    cutoff_time = pd.DataFrame({'time': times, 'instance_id': range(17)})
    property_feature = IdentityFeature(entityset['log']['value']) > 10

    with cluster() as (scheduler, [a, b]):
        dkwargs = {'cluster': scheduler['address']}
        feature_matrix = calculate_feature_matrix([property_feature],
                                                  entityset=entityset,
                                                  cutoff_time=cutoff_time,
                                                  verbose=True,
                                                  chunk_size=.13,
                                                  dask_kwargs=dkwargs,
                                                  approximate='1 hour')

    assert (feature_matrix[property_feature.get_name()] == labels).values.all()


def test_dask_persisted_entityset(entityset, capsys):
    times = list([datetime(2011, 4, 9, 10, 30, i * 6) for i in range(5)] +
                 [datetime(2011, 4, 9, 10, 31, i * 9) for i in range(4)] +
                 [datetime(2011, 4, 9, 10, 40, 0)] +
                 [datetime(2011, 4, 10, 10, 40, i) for i in range(2)] +
                 [datetime(2011, 4, 10, 10, 41, i * 3) for i in range(3)] +
                 [datetime(2011, 4, 10, 11, 10, i * 3) for i in range(2)])
    labels = [False] * 3 + [True] * 2 + [False] * 9 + [True] + [False] * 2
    cutoff_time = pd.DataFrame({'time': times, 'instance_id': range(17)})
    property_feature = IdentityFeature(entityset['log']['value']) > 10

    with cluster() as (scheduler, [a, b]):
        dkwargs = {'cluster': scheduler['address']}
        feature_matrix = calculate_feature_matrix([property_feature],
                                                  entityset=entityset,
                                                  cutoff_time=cutoff_time,
                                                  verbose=True,
                                                  chunk_size=.13,
                                                  dask_kwargs=dkwargs,
                                                  approximate='1 hour')
        assert (feature_matrix[property_feature.get_name()] == labels).values.all()
        feature_matrix = calculate_feature_matrix([property_feature],
                                                  entityset=entityset,
                                                  cutoff_time=cutoff_time,
                                                  verbose=True,
                                                  chunk_size=.13,
                                                  dask_kwargs=dkwargs,
                                                  approximate='1 hour')
        captured = capsys.readouterr()
        assert "Using EntitySet persisted on the cluster as dataset " in captured[0]
        assert (feature_matrix[property_feature.get_name()] == labels).values.all()


class TestCreateClientAndCluster(object):
    def test_user_cluster_as_string(self, entityset, monkeypatch):
        monkeypatch.setitem(create_client_and_cluster.__globals__,
                            'LocalCluster',
                            mock_cluster)
        monkeypatch.setitem(create_client_and_cluster.__globals__,
                            'Client',
                            MockClient)

        # cluster in dask_kwargs case
        client, cluster = create_client_and_cluster(n_jobs=2,
                                                    num_tasks=3,
                                                    dask_kwargs={'cluster': 'tcp://127.0.0.1:54321'},
                                                    entityset_size=1)
        assert cluster == 'tcp://127.0.0.1:54321'

    def test_cluster_creation(self, entityset, monkeypatch):
        total_memory = psutil.virtual_memory().total
        monkeypatch.setitem(create_client_and_cluster.__globals__,
                            'LocalCluster',
                            mock_cluster)
        monkeypatch.setitem(create_client_and_cluster.__globals__,
                            'Client',
                            MockClient)
        try:
            cpus = len(psutil.Process().cpu_affinity())
        except AttributeError:
            cpus = psutil.cpu_count()

        # jobs < tasks case
        client, cluster = create_client_and_cluster(n_jobs=2,
                                                    num_tasks=3,
                                                    dask_kwargs={},
                                                    entityset_size=1)
        num_workers = min(cpus, 2)
        memory_limit = int(total_memory / float(num_workers))
        assert cluster == (min(cpus, 2), 1, None, memory_limit)
        # jobs > tasks case
        client, cluster = create_client_and_cluster(n_jobs=10,
                                                    num_tasks=3,
                                                    dask_kwargs={'diagnostics_port': 8789},
                                                    entityset_size=1)
        num_workers = min(cpus, 3)
        memory_limit = int(total_memory / float(num_workers))
        assert cluster == (num_workers, 1, 8789, memory_limit)

        # dask_kwargs sets memory limit
        client, cluster = create_client_and_cluster(n_jobs=2,
                                                    num_tasks=3,
                                                    dask_kwargs={'diagnostics_port': 8789,
                                                                 'memory_limit': 1000},
                                                    entityset_size=1)
        num_workers = min(cpus, 2)
        assert cluster == (num_workers, 1, 8789, 1000)

    def test_not_enough_memory(self, entityset, monkeypatch):
        total_memory = psutil.virtual_memory().total
        monkeypatch.setitem(create_client_and_cluster.__globals__,
                            'LocalCluster',
                            mock_cluster)
        monkeypatch.setitem(create_client_and_cluster.__globals__,
                            'Client',
                            MockClient)
        # errors if not enough memory for each worker to store the entityset
        with pytest.raises(ValueError, match=''):
            create_client_and_cluster(n_jobs=1,
                                      num_tasks=5,
                                      dask_kwargs={},
                                      entityset_size=total_memory * 2)

        # does not error even if worker memory is less than 2x entityset size
        create_client_and_cluster(n_jobs=1,
                                  num_tasks=5,
                                  dask_kwargs={},
                                  entityset_size=total_memory * .75)


def test_parallel_failure_raises_correct_error(entityset):
    times = list([datetime(2011, 4, 9, 10, 30, i * 6) for i in range(5)] +
                 [datetime(2011, 4, 9, 10, 31, i * 9) for i in range(4)] +
                 [datetime(2011, 4, 9, 10, 40, 0)] +
                 [datetime(2011, 4, 10, 10, 40, i) for i in range(2)] +
                 [datetime(2011, 4, 10, 10, 41, i * 3) for i in range(3)] +
                 [datetime(2011, 4, 10, 11, 10, i * 3) for i in range(2)])
    cutoff_time = pd.DataFrame({'time': times, 'instance_id': range(17)})
    property_feature = IdentityFeature(entityset['log']['value']) > 10

    error_text = 'Need at least one worker'
    with pytest.raises(AssertionError, match=error_text):
        calculate_feature_matrix([property_feature],
                                 entityset=entityset,
                                 cutoff_time=cutoff_time,
                                 verbose=True,
                                 chunk_size=.13,
                                 n_jobs=0,
                                 approximate='1 hour')


def test_n_jobs(entityset):
    try:
        cpus = len(psutil.Process().cpu_affinity())
    except AttributeError:
        cpus = psutil.cpu_count()

    assert n_jobs_to_workers(1) == 1
    assert n_jobs_to_workers(-1) == cpus
    assert n_jobs_to_workers(cpus) == cpus
    assert n_jobs_to_workers((cpus + 1) * -1) == 1
    if cpus > 1:
        assert n_jobs_to_workers(-2) == cpus - 1

    error_text = 'Need at least one worker'
    with pytest.raises(AssertionError, match=error_text):
        n_jobs_to_workers(0)


def test_integer_time_index(int_es):
    times = list(range(8, 18)) + list(range(19, 26))
    labels = [False] * 3 + [True] * 2 + [False] * 9 + [True] + [False] * 2
    cutoff_df = pd.DataFrame({'time': times, 'instance_id': range(17)})
    property_feature = IdentityFeature(int_es['log']['value']) > 10

    feature_matrix = calculate_feature_matrix([property_feature],
                                              int_es,
                                              cutoff_time=cutoff_df,
                                              cutoff_time_in_index=True)

    time_level_vals = feature_matrix.index.get_level_values(1).values
    sorted_df = cutoff_df.sort_values(['time', 'instance_id'], kind='mergesort')
    assert (time_level_vals == sorted_df['time'].values).all()
    assert (feature_matrix[property_feature.get_name()] == labels).values.all()


def test_integer_time_index_datetime_cutoffs(int_es):
    times = [datetime.now()] * 17
    cutoff_df = pd.DataFrame({'time': times, 'instance_id': range(17)})
    property_feature = IdentityFeature(int_es['log']['value']) > 10

    error_text = "cutoff_time times must be numeric: try casting via "\
        "pd\\.to_numeric\\(cutoff_time\\['time'\\]\\)"
    with pytest.raises(TypeError, match=error_text):
        calculate_feature_matrix([property_feature],
                                 int_es,
                                 cutoff_time=cutoff_df,
                                 cutoff_time_in_index=True)


def test_integer_time_index_passes_extra_columns(int_es):
    times = list(range(8, 18)) + list(range(19, 23)) + [25, 24, 23]
    labels = [False] * 3 + [True] * 2 + [False] * 9 + [False] * 2 + [True]
    instances = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 16, 15, 14]
    cutoff_df = pd.DataFrame({'time': times,
                              'instance_id': instances,
                              'labels': labels})
    cutoff_df = cutoff_df[['time', 'instance_id', 'labels']]
    property_feature = IdentityFeature(int_es['log']['value']) > 10

    fm = calculate_feature_matrix([property_feature],
                                  int_es,
                                  cutoff_time=cutoff_df,
                                  cutoff_time_in_index=True)

    assert (fm[property_feature.get_name()] == fm['labels']).all()


def test_integer_time_index_mixed_cutoff(int_es):
    times_dt = list(range(8, 17)) + [datetime(2011, 1, 1), 19, 20, 21, 22, 25, 24, 23]
    labels = [False] * 3 + [True] * 2 + [False] * 9 + [False] * 2 + [True]
    instances = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 16, 15, 14]
    cutoff_df = pd.DataFrame({'time': times_dt,
                              'instance_id': instances,
                              'labels': labels})
    cutoff_df = cutoff_df[['time', 'instance_id', 'labels']]
    property_feature = IdentityFeature(int_es['log']['value']) > 10

    error_text = 'cutoff_time times must be.*try casting via.*'
    with pytest.raises(TypeError, match=error_text):
        calculate_feature_matrix([property_feature],
                                 int_es,
                                 cutoff_time=cutoff_df)

    times_str = list(range(8, 17)) + ["foobar", 19, 20, 21, 22, 25, 24, 23]
    cutoff_df['time'] = times_str
    with pytest.raises(TypeError, match=error_text):
        calculate_feature_matrix([property_feature],
                                 int_es,
                                 cutoff_time=cutoff_df)

    times_date_str = list(range(8, 17)) + ['2018-04-02', 19, 20, 21, 22, 25, 24, 23]
    cutoff_df['time'] = times_date_str
    with pytest.raises(TypeError, match=error_text):
        calculate_feature_matrix([property_feature],
                                 int_es,
                                 cutoff_time=cutoff_df)

    [19, 20, 21, 22]
    times_int_str = [0, 1, 2, 3, 4, 5, '6', 7, 8, 9, 9, 10, 11, 12, 15, 14, 13]
    times_int_str = list(range(8, 17)) + ['17', 19, 20, 21, 22, 25, 24, 23]
    cutoff_df['time'] = times_int_str
    # calculate_feature_matrix should convert time column to ints successfully here
    with pytest.raises(TypeError, match=error_text):
        calculate_feature_matrix([property_feature],
                                 int_es,
                                 cutoff_time=cutoff_df)


def test_datetime_index_mixed_cutoff(entityset):
    times = list([datetime(2011, 4, 9, 10, 30, i * 6) for i in range(5)] +
                 [datetime(2011, 4, 9, 10, 31, i * 9) for i in range(4)] +
                 [17] +
                 [datetime(2011, 4, 10, 10, 40, i) for i in range(2)] +
                 [datetime(2011, 4, 10, 10, 41, i * 3) for i in range(3)] +
                 [datetime(2011, 4, 10, 11, 10, i * 3) for i in range(2)])
    labels = [False] * 3 + [True] * 2 + [False] * 9 + [False] * 2 + [True]
    instances = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 16, 15, 14]
    cutoff_df = pd.DataFrame({'time': times,
                              'instance_id': instances,
                              'labels': labels})
    cutoff_df = cutoff_df[['time', 'instance_id', 'labels']]
    property_feature = IdentityFeature(entityset['log']['value']) > 10

    error_text = 'cutoff_time times must be.*try casting via.*'
    with pytest.raises(TypeError, match=error_text):
        calculate_feature_matrix([property_feature],
                                 entityset,
                                 cutoff_time=cutoff_df)

    times[9] = "foobar"
    cutoff_df['time'] = times
    with pytest.raises(TypeError, match=error_text):
        calculate_feature_matrix([property_feature],
                                 entityset,
                                 cutoff_time=cutoff_df)

    cutoff_df['time'].iloc[9] = '2018-04-02 18:50:45.453216'
    with pytest.raises(TypeError, match=error_text):
        calculate_feature_matrix([property_feature],
                                 entityset,
                                 cutoff_time=cutoff_df)

    times[9] = '17'
    cutoff_df['time'] = times
    with pytest.raises(TypeError, match=error_text):
        calculate_feature_matrix([property_feature],
                                 entityset,
                                 cutoff_time=cutoff_df)


def test_string_time_values_in_cutoff_time(entityset):
    times = ['2011-04-09 10:31:27', '2011-04-09 10:30:18']
    cutoff_time = pd.DataFrame({'time': times, 'instance_id': [0, 0]})
    agg_feature = ft.Feature(entityset['log']['value'], parent_entity=entityset['customers'], primitive=Sum)

    error_text = 'cutoff_time times must be.*try casting via.*'
    with pytest.raises(TypeError, match=error_text):
        calculate_feature_matrix([agg_feature], entityset, cutoff_time=cutoff_time)


def test_no_data_for_cutoff_time():
    es = ft.demo.load_mock_customer(return_entityset=True, random_seed=0)
    cutoff_times = pd.DataFrame({"customer_id": [4],
                                 "time": pd.Timestamp('2011-04-08 20:08:13')})

    trans_per_session = ft.Feature(es["transactions"]["transaction_id"], parent_entity=es["sessions"], primitive=Count)
    trans_per_customer = ft.Feature(es["transactions"]["transaction_id"], parent_entity=es["customers"], primitive=Count)
    features = [trans_per_customer, ft.Feature(trans_per_session, parent_entity=es["customers"], primitive=Max)]

    fm = ft.calculate_feature_matrix(features, entityset=es, cutoff_time=cutoff_times)

    # due to default values for each primitive
    # count will be 0, but max will nan
    np.testing.assert_array_equal(fm.values, [[0, np.nan]])
