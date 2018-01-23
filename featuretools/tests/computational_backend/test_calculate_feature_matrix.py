import copy
import os
import shutil
from builtins import range
from datetime import datetime
from random import randint

import numpy as np
import pandas as pd
import pytest

from ..testing_utils import make_ecommerce_entityset

from featuretools import EntitySet, Timedelta, calculate_feature_matrix, dfs
from featuretools.computational_backends.calculate_feature_matrix import (
    bin_cutoff_times
)
from featuretools.primitives import (
    AggregationPrimitive,
    Count,
    DirectFeature,
    IdentityFeature,
    Min,
    Sum
)


@pytest.fixture(scope='module')
def entityset():
    return make_ecommerce_entityset()


# TODO test mean ignores nan values
def test_calc_feature_matrix(entityset):
    times = list([datetime(2011, 4, 9, 10, 30, i * 6) for i in range(5)] +
                 [datetime(2011, 4, 9, 10, 31, i * 9) for i in range(4)] +
                 [datetime(2011, 4, 9, 10, 40, 0)] +
                 [datetime(2011, 4, 10, 10, 40, i) for i in range(2)] +
                 [datetime(2011, 4, 10, 10, 41, i * 3) for i in range(3)] +
                 [datetime(2011, 4, 10, 11, 10, i * 3) for i in range(2)])
    labels = [False] * 3 + [True] * 2 + [False] * 9 + [True] + [False] * 2

    property_feature = IdentityFeature(entityset['log']['value']) > 10

    feature_matrix = calculate_feature_matrix([property_feature], instance_ids=range(17),
                                              cutoff_time=times, verbose=True)

    assert (feature_matrix == labels).values.all()

    with pytest.raises(AssertionError):
        feature_matrix = calculate_feature_matrix('features', instance_ids=range(17),
                                                  cutoff_time=times)
    with pytest.raises(AssertionError):
        feature_matrix = calculate_feature_matrix([], instance_ids=range(17),
                                                  cutoff_time=times)
    with pytest.raises(AssertionError):
        feature_matrix = calculate_feature_matrix([1, 2, 3], instance_ids=range(17),
                                                  cutoff_time=times)


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
                                  AggregationPrimitive)]
    property_feature = IdentityFeature(es['trips']['trip_id'])
    # direct_agg_feat = DirectFeature(Sum(es['trips']['trip_duration'],
    #                                     es['flights']),
    #                                 es['trips'])
    cutoff_time = pd.DataFrame.from_dict({'instance_id': df['trip_id'],
                                          'time': df['flight_time']})
    time_feature = IdentityFeature(es['trips']['flight_time'])
    feature_matrix = calculate_feature_matrix(flight_features + [property_feature, time_feature],
                                              cutoff_time_in_index=True,
                                              cutoff_time=cutoff_time)
    feature_matrix.index.names = ['instance', 'time']
    assert(np.all(feature_matrix.reset_index('time').reset_index()[['instance', 'time']].values == feature_matrix[['trip_id', 'flight_time']].values))
    feature_matrix_2 = calculate_feature_matrix(flight_features + [property_feature, time_feature],
                                                cutoff_time=cutoff_time,
                                                cutoff_time_in_index=True,
                                                approximate=Timedelta(2, 'd'))
    feature_matrix_2.index.names = ['instance', 'time']
    assert(np.all(feature_matrix_2.reset_index('time').reset_index()[['instance', 'time']].values == feature_matrix_2[['trip_id', 'flight_time']].values))
    for column in feature_matrix:
        for x, y in zip(feature_matrix[column], feature_matrix_2[column]):
            if not ((pd.isnull(x) and pd.isnull(y)) or (x == y)):
                import pdb
                pdb.set_trace()
            assert ((pd.isnull(x) and pd.isnull(y)) or (x == y))


def test_cfm_no_cutoff_time_index(entityset):
    es = entityset
    agg_feat = Count(es['log']['id'], es['sessions'])
    agg_feat4 = Sum(agg_feat, es['customers'])
    dfeat = DirectFeature(agg_feat4, es['sessions'])
    feature_matrix = calculate_feature_matrix([dfeat, agg_feat],
                                              instance_ids=[0, 2],
                                              cutoff_time_in_index=False,
                                              approximate=Timedelta(12, 's'),
                                              cutoff_time=[datetime(2013, 4, 9, 10, 31, 19),
                                                           datetime(2013, 4, 9, 11, 0, 0)])
    assert feature_matrix.index.name == 'id'
    assert feature_matrix.index.values.tolist() == [0, 2]
    assert feature_matrix[dfeat.get_name()].tolist() == [10, 10]
    assert feature_matrix[agg_feat.get_name()].tolist() == [5, 1]
    feature_matrix_2 = calculate_feature_matrix([dfeat, agg_feat],
                                                instance_ids=[0, 2],
                                                cutoff_time_in_index=False,
                                                approximate=Timedelta(10, 's'),
                                                cutoff_time=[datetime(2011, 4, 9, 10, 31, 19),
                                                             datetime(2011, 4, 9, 11, 0, 0)])
    assert feature_matrix_2.index.name == 'id'
    assert feature_matrix_2.index.tolist() == [0, 2]
    assert feature_matrix_2[dfeat.get_name()].tolist() == [7, 10]
    assert feature_matrix_2[agg_feat.get_name()].tolist() == [5, 1]


def test_saveprogress(entityset):
    times = list([datetime(2011, 4, 9, 10, 30, i * 6) for i in range(5)] +
                 [datetime(2011, 4, 9, 10, 31, i * 9) for i in range(4)] +
                 [datetime(2011, 4, 9, 10, 40, 0)] +
                 [datetime(2011, 4, 10, 10, 40, i) for i in range(2)] +
                 [datetime(2011, 4, 10, 10, 41, i * 3) for i in range(3)] +
                 [datetime(2011, 4, 10, 11, 10, i * 3) for i in range(2)])
    property_feature = IdentityFeature(entityset['log']['value']) > 10
    save_progress = os.path.join(os.path.expanduser('~'), 'ft_temp')
    if not os.path.exists(save_progress):
        os.makedirs(save_progress)
    if len(os.listdir(save_progress)) > 0:
        for file_path in os.listdir(save_progress):
            os.remove(os.path.join(save_progress, file_path))
    fm_save = calculate_feature_matrix([property_feature],
                                       instance_ids=range(17),
                                       cutoff_time=times,
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
    merged_df.set_index(pd.DatetimeIndex(times, append=True, inplace=True))
    fm_no_save = calculate_feature_matrix([property_feature],
                                          instance_ids=range(17),
                                          cutoff_time=times)
    assert np.all((merged_df.sort_index().values) == (fm_save.sort_index().values))
    assert np.all((fm_no_save.sort_index().values) == (fm_save.sort_index().values))
    assert np.all((fm_no_save.sort_index().values) == (merged_df.sort_index().values))
    shutil.rmtree(save_progress)


def test_cutoff_time_correctly(entityset):
    property_feature = Count(entityset['log']['id'], entityset['customers'])
    feature_matrix = calculate_feature_matrix([property_feature], instance_ids=[0, 1, 2],
                                              cutoff_time=[datetime(2011, 4, 10), datetime(2011, 4, 11),
                                                           datetime(2011, 4, 7)])
    labels = [0, 10, 5]
    assert (feature_matrix == labels).values.all()


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
    property_feature = Count(entityset['log']['id'], entityset['customers'])
    top_level_agg = Count(entityset['customers']['id'], entityset['regions'])

    # make sure features that have a direct to a higher level agg
    # so we have multiple "filter eids" in get_pandas_data_slice,
    # and we go through the loop to pull data with a training_window param more than once
    dagg = DirectFeature(top_level_agg, entityset['customers'])

    # for now, warns if last_time_index not present
    feature_matrix = calculate_feature_matrix([property_feature, dagg],
                                              instance_ids=[0, 1, 2],
                                              cutoff_time=[datetime(2011, 4, 9, 12, 31),
                                                           datetime(2011, 4, 10, 11),
                                                           datetime(2011, 4, 10, 13, 10, 1)],
                                              training_window='2 hours')

    entityset.add_last_time_indexes()

    with pytest.raises(AssertionError):
        feature_matrix = calculate_feature_matrix([property_feature],
                                                  instance_ids=[0, 1, 2],
                                                  cutoff_time=[datetime(2011, 4, 9, 12, 31),
                                                               datetime(2011, 4, 10, 11),
                                                               datetime(2011, 4, 10, 13, 10, 1)],
                                                  training_window=Timedelta(2, 'observations', entity='log'))

    feature_matrix = calculate_feature_matrix([property_feature, dagg],
                                              instance_ids=[0, 1, 2, 4],
                                              cutoff_time=[datetime(2011, 4, 9, 12, 31),
                                                           datetime(2011, 4, 10, 11),
                                                           datetime(2011, 4, 10, 13, 10, 1)],
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
        'region_id': ['United States'],
        'cohort': [1],
        'cohort_name': ["Late Adopters"],
        'loves_ice_cream': [True],
        'favorite_quote': ["Who is John Galt?"],
        'signup_date': [datetime(2011, 4, 10)],
        'upgrade_date': [datetime(2011, 4, 12)],
        'cancel_date': [datetime(2011, 5, 13)],
        'date_of_birth': [datetime(1938, 2, 1)],
        'engagement_level': [2],
    }
    df = pd.DataFrame(row)
    df.index = range(3, 4)
    df = entityset['customers'].df.append(df)
    entityset['customers'].update_data(df)
    entityset.add_last_time_indexes()

    property_feature = Count(entityset['log']['id'], entityset['customers'])
    top_level_agg = Count(entityset['customers']['id'], entityset['regions'])
    dagg = DirectFeature(top_level_agg, entityset['customers'])

    feature_matrix = calculate_feature_matrix(
        [property_feature, dagg],
        instance_ids=[0, 1, 2, 3],
        cutoff_time=[datetime(2011, 4, 9, 12, 31),
                     datetime(2011, 4, 10, 11),
                     datetime(2011, 4, 10, 13, 10, 1),
                     datetime(2011, 4, 10, 1, 59, 59)],
        training_window='2 hours'
    )
    prop_values = [5, 5, 1, 0]
    dagg_values = [3, 2, 1, 2]
    feature_matrix.sort_index(inplace=True)
    assert (feature_matrix[property_feature.get_name()] == prop_values).values.all()
    assert (feature_matrix[dagg.get_name()] == dagg_values).values.all()


def test_approximate_multiple_instances_per_cutoff_time(entityset):
    es = entityset
    agg_feat = Count(es['log']['id'], es['sessions'])
    agg_feat2 = Sum(agg_feat, es['customers'])
    dfeat = DirectFeature(agg_feat2, es['sessions'])

    feature_matrix = calculate_feature_matrix([dfeat, agg_feat],
                                              instance_ids=[0, 2],
                                              approximate=Timedelta(1, 'week'),
                                              cutoff_time=[datetime(2011, 4, 9, 10, 31, 19),
                                                           datetime(2011, 4, 9, 11, 0, 0)])
    assert feature_matrix.shape[0] == 2
    assert feature_matrix[dfeat.get_name()].dropna().shape[0] == 0
    assert feature_matrix[agg_feat.get_name()].tolist() == [5, 1]


def test_approximate_dfeat_of_agg_on_target(entityset):
    es = entityset
    agg_feat = Count(es['log']['id'], es['sessions'])
    agg_feat2 = Sum(agg_feat, es['customers'])
    dfeat = DirectFeature(agg_feat2, es['sessions'])

    feature_matrix = calculate_feature_matrix([dfeat, agg_feat],
                                              instance_ids=[0, 2],
                                              approximate=Timedelta(10, 's'),
                                              cutoff_time=[datetime(2011, 4, 9, 10, 31, 19),
                                                           datetime(2011, 4, 9, 11, 0, 0)])
    assert feature_matrix[dfeat.get_name()].tolist() == [7, 10]
    assert feature_matrix[agg_feat.get_name()].tolist() == [5, 1]


def test_approximate_dfeat_of_dfeat_of_agg_on_target(entityset):
    es = entityset
    agg_feat = Count(es['log']['id'], es['sessions'])
    agg_feat2 = Sum(agg_feat, es['customers'])
    dfeat = DirectFeature(agg_feat2, es['log'])

    feature_matrix = calculate_feature_matrix([dfeat],
                                              instance_ids=[0, 2],
                                              approximate=Timedelta(10, 's'),
                                              cutoff_time=[datetime(2011, 4, 9, 10, 31, 19),
                                                           datetime(2011, 4, 9, 11, 0, 0)])
    assert feature_matrix[dfeat.get_name()].tolist() == [7, 10]


def test_empty_path_approximate_full(entityset):
    es = copy.deepcopy(entityset)
    es['sessions'].df['customer_id'] = [np.nan, np.nan, np.nan, 1, 1, 2]
    agg_feat = Count(es['log']['id'], es['sessions'])
    agg_feat2 = Sum(agg_feat, es['customers'])
    dfeat = DirectFeature(agg_feat2, es['sessions'])

    feature_matrix = calculate_feature_matrix([dfeat, agg_feat],
                                              instance_ids=[0, 2],
                                              approximate=Timedelta(10, 's'),
                                              cutoff_time=[datetime(2011, 4, 9, 10, 31, 19),
                                                           datetime(2011, 4, 9, 11, 0, 0)])
    vals1 = feature_matrix[dfeat.get_name()].tolist()
    assert np.isnan(vals1[0])
    assert np.isnan(vals1[1])
    assert feature_matrix[agg_feat.get_name()].tolist() == [5, 1]


def test_empty_path_approximate_partial(entityset):
    es = copy.deepcopy(entityset)
    es['sessions'].df['customer_id'] = [0, 0, np.nan, 1, 1, 2]
    agg_feat = Count(es['log']['id'], es['sessions'])
    agg_feat2 = Sum(agg_feat, es['customers'])
    dfeat = DirectFeature(agg_feat2, es['sessions'])

    feature_matrix = calculate_feature_matrix([dfeat, agg_feat],
                                              instance_ids=[0, 2],
                                              approximate=Timedelta(10, 's'),
                                              cutoff_time=[datetime(2011, 4, 9, 10, 31, 19),
                                                           datetime(2011, 4, 9, 11, 0, 0)])
    vals1 = feature_matrix[dfeat.get_name()].tolist()
    assert vals1[0] == 7
    assert np.isnan(vals1[1])
    assert feature_matrix[agg_feat.get_name()].tolist() == [5, 1]


def test_approx_base_feature_is_also_first_class_feature(entityset):
    es = entityset
    log_to_products = DirectFeature(es['products']['rating'], es['log'])
    # This should still be computed properly
    agg_feat = Min(log_to_products, es['sessions'])
    customer_agg_feat = Sum(agg_feat, es['customers'])
    # This is to be approximated
    sess_to_cust = DirectFeature(customer_agg_feat, es['sessions'])

    feature_matrix = calculate_feature_matrix([sess_to_cust, agg_feat],
                                              instance_ids=[0, 2],
                                              approximate=Timedelta(10, 's'),
                                              cutoff_time=[datetime(2011, 4, 9, 10, 31, 19),
                                                           datetime(2011, 4, 9, 11, 0, 0)])
    vals1 = feature_matrix[sess_to_cust.get_name()].tolist()
    assert vals1 == [8.5, 7]
    vals2 = feature_matrix[agg_feat.get_name()].tolist()
    assert vals2 == [4, 1.5]


def test_approximate_time_split_returns_the_same_result(entityset):
    es = entityset
    agg_feat = Count(es['log']['id'], es['sessions'])
    agg_feat2 = Sum(agg_feat, es['customers'])
    dfeat = DirectFeature(agg_feat2, es['sessions'])

    cutoff_df = pd.DataFrame({'time': [pd.Timestamp('2011-04-09 10:07:30'),
                                       pd.Timestamp('2011-04-09 10:07:40')],
                              'instance_id': [0, 0]})

    feature_matrix_at_once = calculate_feature_matrix([dfeat, agg_feat],
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
    agg_feat = Count(es['log']['id'], es['customers'])
    dfeat = DirectFeature(agg_feat, es['sessions'])

    cutoff_df = pd.DataFrame({'time': [pd.Timestamp('2011-04-08 11:00:00'),
                                       pd.Timestamp('2011-04-09 11:00:00')],
                              'instance_id': [0, 0]})

    fm = calculate_feature_matrix([dfeat],
                                  approximate=Timedelta(10, 's'),
                                  cutoff_time=cutoff_df)
    assert fm[dfeat.get_name()].tolist() == [0, 10]


# def test_approximate_deep_recurse(entityset):
    # es = entityset
    # agg_feat = Count(es['customers']['id'], es['regions'])
    # dfeat1 = DirectFeature(agg_feat, es['sessions'])
    # agg_feat2 = Sum(dfeat1, es['customers'])
    # dfeat2 = DirectFeature(agg_feat2, es['sessions'])

    # agg_feat3 = Count(es['log']['id'], es['products'])
    # dfeat3 = DirectFeature(agg_feat3, es['log'])
    # agg_feat4 = Sum(dfeat3, es['sessions'])

    # feature_matrix = calculate_feature_matrix([dfeat2, agg_feat4],
    #                                          instance_ids=[0, 2],
    #                                          approximate=Timedelta(10, 's'),
    #                                          cutoff_time=[datetime(2011, 4, 9, 10, 31, 19),
    #                                                       datetime(2011, 4, 9, 11, 0, 0)])
    # # dfeat2 and agg_feat4 should both be approximated


def test_approximate_child_aggs_handled_correctly(entityset):
    es = entityset
    agg_feat = Count(es['customers']['id'], es['regions'])
    dfeat = DirectFeature(agg_feat, es['customers'])
    agg_feat_2 = Count(es['log']['value'], es['customers'])
    cutoff_df = pd.DataFrame({'time': [pd.Timestamp('2011-04-08 10:30:00'),
                                       pd.Timestamp('2011-04-09 10:30:06')],
                              'instance_id': [0, 0]})

    fm = calculate_feature_matrix([dfeat],
                                  approximate=Timedelta(10, 's'),
                                  cutoff_time=cutoff_df)
    fm_2 = calculate_feature_matrix([dfeat, agg_feat_2],
                                    approximate=Timedelta(10, 's'),
                                    cutoff_time=cutoff_df)
    assert fm[dfeat.get_name()].tolist() == [2, 3]
    assert fm_2[agg_feat_2.get_name()].tolist() == [0, 2]


def test_cutoff_time_naming(entityset):
    es = entityset

    agg_feat = Count(es['customers']['id'], es['regions'])
    dfeat = DirectFeature(agg_feat, es['customers'])
    cutoff_df = pd.DataFrame({'time': [pd.Timestamp('2011-04-08 10:30:00'),
                                       pd.Timestamp('2011-04-09 10:30:06')],
                              'instance_id': [0, 0]})
    cutoff_df_index_name = cutoff_df.rename(columns={"instance_id": "id"})
    cutoff_df_time_name = cutoff_df.rename(columns={"time": "cutoff_time"})
    cutoff_df_index_name_time_name = cutoff_df.rename(columns={"instance_id": "id", "time": "cutoff_time"})
    cutoff_df_wrong_index_name = cutoff_df.rename(columns={"instance_id": "wrong_id"})

    fm1 = calculate_feature_matrix([dfeat], cutoff_time=cutoff_df)
    for test_cutoff in [cutoff_df_index_name, cutoff_df_time_name, cutoff_df_index_name_time_name]:
        fm2 = calculate_feature_matrix([dfeat], cutoff_time=test_cutoff)

        assert all((fm1 == fm2.values).values)

    with pytest.raises(AttributeError):
        calculate_feature_matrix([dfeat], cutoff_time=cutoff_df_wrong_index_name)


def test_cutoff_time_extra_columns(entityset):
    es = entityset

    agg_feat = Count(es['customers']['id'], es['regions'])
    dfeat = DirectFeature(agg_feat, es['customers'])

    cutoff_df = pd.DataFrame({'time': [pd.Timestamp('2011-04-09 10:30:06'),
                                       pd.Timestamp('2011-04-08 10:30:00')],
                              'instance_id': [0, 0],
                              'label': [True, False]},
                             columns=['time', 'instance_id', 'label'])
    fm = calculate_feature_matrix([dfeat], cutoff_time=cutoff_df)
    # check column was added to end of matrix
    assert 'label' == fm.columns[-1]
    # check column was sorted by time labelike the rest of the feature matrix
    true_series = pd.Series([False, True], index=[0, 0])
    assert (fm['label'] == true_series).all()

    fm_2 = calculate_feature_matrix([dfeat],
                                    cutoff_time=cutoff_df,
                                    approximate="2 days")
    # check column was added to end of matrix
    assert 'label' in fm_2.columns
    # check column was sorted by time like the rest of the feature matrix
    true_series = pd.Series([False, True], index=[0, 0])
    assert (fm_2['label'] == true_series).all()


def test_approximate_returns_original_time_indexes(entityset):
    es = entityset

    agg_feat = Count(es['customers']['id'], es['regions'])
    dfeat = DirectFeature(agg_feat, es['customers'])
    cutoff_df = pd.DataFrame({'time': [pd.Timestamp('2011-04-09 10:30:06'),
                                       pd.Timestamp('2011-04-08 10:30:00')],
                              'instance_id': [0, 0]})

    fm = calculate_feature_matrix([dfeat],
                                  cutoff_time=cutoff_df,
                                  approximate="2 days",
                                  cutoff_time_in_index=True)
    instance_level_vals = fm.index.get_level_values(0).values
    time_level_vals = fm.index.get_level_values(1).values
    cutoff_df.sort_values(['time'], inplace=True, kind='mergesort')
    assert (instance_level_vals == cutoff_df['instance_id'].values).all()
    assert (time_level_vals == cutoff_df['time'].values).all()
