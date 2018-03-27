from __future__ import division

import gc
import logging
import os
import shutil
from builtins import zip
from collections import defaultdict
from datetime import datetime
from functools import wraps

import numpy as np
import pandas as pd
from pandas.tseries.frequencies import to_offset

from .pandas_backend import PandasBackend

from featuretools.primitives import (
    AggregationPrimitive,
    DirectFeature,
    PrimitiveBase
)
from featuretools.utils.gen_utils import make_tqdm_iterator
from featuretools.utils.wrangle import _check_timedelta

logger = logging.getLogger('featuretools.computational_backend')


def calculate_feature_matrix(features, cutoff_time=None, instance_ids=None,
                             entities=None, relationships=None, entityset=None,
                             cutoff_time_in_index=False,
                             training_window=None, approximate=None,
                             save_progress=None, verbose=False,
                             backend_verbose=False,
                             profile=False):
    """Calculates a matrix for a given set of instance ids and calculation times.

    Args:
        features (list[PrimitiveBase]): Feature definitions to be calculated.

        cutoff_time (pd.DataFrame or Datetime): Specifies at which time to calculate
            the features for each instance.  Can either be a DataFrame with
            'instance_id' and 'time' columns, DataFrame with the name of the
            index variable in the target entity and a time column, a list of values, or a single
            value to calculate for all instances. If the dataframe has more than two columns, any additional
            columns will be added to the resulting feature matrix.

        instance_ids (list): List of instances to calculate features on. Only
            used if cutoff_time is a single datetime.

        entities (dict[str -> tuple(pd.DataFrame, str, str)]): dictionary of
            entities. Entries take the format
            {entity id: (dataframe, id column, (time_column))}.

        relationships (list[(str, str, str, str)]): list of relationships
            between entities. List items are a tuple with the format
            (parent entity id, parent variable, child entity id, child variable).

        entityset (EntitySet): An already initialized entityset. Required if
            entities and relationships are not defined.

        cutoff_time_in_index (bool): If True, return a DataFrame with a MultiIndex
            where the second index is the cutoff time (first is instance id).
            DataFrame will be sorted by (time, instance_id).

        training_window (dict[str -> Timedelta] or Timedelta, optional):
            Window or windows defining how much older than the cutoff time data
            can be to be included when calculating the feature.  To specify
            which entities to apply windows to, use a dictionary mapping entity
            id -> Timedelta. If None, all older data is used.

        approximate (Timedelta or str): Frequency to group instances with similar
            cutoff times by for features with costly calculations. For example,
            if bucket is 24 hours, all instances with cutoff times on the same
            day will use the same calculation for expensive features.

        verbose (bool, optional): Print progress info. The time granularity is per time group
            unless there is only a single cutoff time, in which case backend_verbose is turned on

        backend_verbose (bool, optional): Print progress info of each feature calculatation step per time group.

        profile (bool, optional): Enables profiling if True.

        save_progress (str, optional): path to save intermediate computational results.
    """
    assert (isinstance(features, list) and features != [] and
            all([isinstance(feature, PrimitiveBase) for feature in features])), \
        "features must be a non-empty list of features"

    # handle loading entityset
    from featuretools.entityset.entityset import EntitySet
    if not isinstance(entityset, EntitySet):
        if entities is not None and relationships is not None:
            entityset = EntitySet("entityset", entities, relationships)

    if entityset is not None:
        for f in features:
            f.entityset = entityset

    entityset = features[0].entityset
    target_entity = features[0].entity
    pass_columns = []

    if not isinstance(cutoff_time, pd.DataFrame):
        if cutoff_time is None:
            cutoff_time = datetime.now()

        if instance_ids is None:
            index_var = target_entity.index
            instance_ids = target_entity.df[index_var].tolist()

        if not isinstance(cutoff_time, list):
            cutoff_time = [cutoff_time] * len(instance_ids)

        map_args = [(id, time) for id, time in zip(instance_ids, cutoff_time)]
        df_args = pd.DataFrame(map_args, columns=['instance_id', 'time'])
        to_calc = df_args.values
        cutoff_time = pd.DataFrame(to_calc, columns=['instance_id', 'time'])
    else:
        cutoff_time = cutoff_time.copy()

        # handle how columns are names in cutoff_time
        if "instance_id" not in cutoff_time.columns:
            if target_entity.index not in cutoff_time.columns:
                raise AttributeError('Name of the index variable in the target entity'
                                     ' or "instance_id" must be present in cutoff_time')
            # rename to instance_id
            cutoff_time.rename(columns={target_entity.index: "instance_id"}, inplace=True)

        if "time" not in cutoff_time.columns:
            # take the first column that isn't instance_id and assume it is time
            not_instance_id = [c for c in cutoff_time.columns if c != "instance_id"]
            cutoff_time.rename(columns={not_instance_id[0]: "time"}, inplace=True)
        pass_columns = [column_name for column_name in cutoff_time.columns[2:]]

    backend = PandasBackend(entityset, features)

    # Get dictionary of features to approximate
    if approximate is not None:
        to_approximate, all_approx_feature_set = gather_approximate_features(features, backend)
    else:
        to_approximate = defaultdict(list)
        all_approx_feature_set = None

    # Check if there are any non-approximated aggregation features
    no_unapproximated_aggs = True
    for feature in features:
        if isinstance(feature, AggregationPrimitive):
            # do not need to check if feature is in to_approximate since
            # only base features of direct features can be in to_approximate
            no_unapproximated_aggs = False
            break

        deps = feature.get_deep_dependencies(all_approx_feature_set)
        for dependency in deps:
            if (isinstance(dependency, AggregationPrimitive) and
                    dependency not in to_approximate[dependency.entity.id]):
                no_unapproximated_aggs = False
                break

    cutoff_df_time_var = 'time'
    target_time = '_original_time'

    if approximate is not None:
        # If there are approximated aggs, bin times
        binned_cutoff_time = bin_cutoff_times(cutoff_time.copy(), approximate)

        # Think about collisions: what if original time is a feature
        binned_cutoff_time[target_time] = cutoff_time[cutoff_df_time_var]

        grouped = binned_cutoff_time.groupby(cutoff_df_time_var, sort=True)

    else:
        grouped = cutoff_time.groupby(cutoff_df_time_var, sort=True)

    # if the backend is going to be verbose, don't make cutoff times verbose
    if verbose and not backend_verbose:
        pbar_string = ("Elapsed: {elapsed} | Remaining: {remaining} | "
                       "Progress: {l_bar}{bar}|| "
                       "Calculated: {n}/{total} cutoff times")
        iterator = make_tqdm_iterator(iterable=grouped,
                                      total=len(grouped),
                                      bar_format=pbar_string,
                                      unit="cutoff time")
    else:
        iterator = grouped

    feature_matrix = []
    for _, group in iterator:
        _feature_matrix = calculate_batch(features, group, approximate,
                                          entityset, backend_verbose,
                                          training_window, profile, verbose,
                                          save_progress, backend,
                                          no_unapproximated_aggs, cutoff_df_time_var,
                                          target_time, pass_columns)
        feature_matrix.append(_feature_matrix)
        # Do a manual garbage collection in case objects from calculate_batch
        # weren't collected automatically
        gc.collect()

    feature_matrix = pd.concat(feature_matrix)
    if not cutoff_time_in_index:
        feature_matrix.reset_index(level='time', drop=True, inplace=True)

    if save_progress and os.path.exists(os.path.join(save_progress, 'temp')):
        shutil.rmtree(os.path.join(save_progress, 'temp'))

    return feature_matrix


def calculate_batch(features, group, approximate, entityset, backend_verbose,
                    training_window, profile, verbose, save_progress, backend,
                    no_unapproximated_aggs, cutoff_df_time_var, target_time,
                    pass_columns):
    # if approximating, calculate the approximate features
    if approximate is not None:
        precalculated_features, all_approx_feature_set = approximate_features(
            features,
            group,
            window=approximate,
            entityset=entityset,
            backend=backend,
            training_window=training_window,
            verbose=backend_verbose,
            profile=profile
        )
    else:
        precalculated_features = None
        all_approx_feature_set = None

    # if backend verbose wasn't set explicitly, set to True if verbose is true
    # and there is only 1 cutoff time
    if backend_verbose is None:
        one_cutoff_time = group[cutoff_df_time_var].nunique() == 1
        backend_verbose = verbose and one_cutoff_time

    @save_csv_decorator(save_progress)
    def calc_results(time_last, ids, precalculated_features=None, training_window=None):
        matrix = backend.calculate_all_features(ids, time_last,
                                                training_window=training_window,
                                                precalculated_features=precalculated_features,
                                                ignored=all_approx_feature_set,
                                                profile=profile,
                                                verbose=backend_verbose)
        return matrix

    # if all aggregations have been approximated, can calculate all together
    if no_unapproximated_aggs and approximate is not None:
        grouped = [[datetime.now(), group]]
    else:
        # if approximated features, set cutoff_time to unbinned time
        if precalculated_features is not None:
            group[cutoff_df_time_var] = group[target_time]

        grouped = group.groupby(cutoff_df_time_var, sort=True)

    feature_matrix = []
    for _time_last_to_calc, group in grouped:
        # sort group by instance id
        ids = group['instance_id'].sort_values().values
        time_last = group[cutoff_df_time_var].iloc[0]
        if no_unapproximated_aggs and approximate is not None:
            window = None
        else:
            window = training_window

        # calculate values for those instances at time _time_last_to_calc
        _feature_matrix = calc_results(_time_last_to_calc,
                                       ids,
                                       precalculated_features=precalculated_features,
                                       training_window=window)

        id_name = _feature_matrix.index.name

        # if approximate, merge feature matrix with group frame to get original
        # cutoff times and passed columns
        if approximate:
            indexer = group[['instance_id', target_time] + pass_columns]
            _feature_matrix = indexer.merge(_feature_matrix,
                                            left_on=['instance_id'],
                                            right_index=True,
                                            how='left')
            _feature_matrix.set_index(['instance_id', target_time], inplace=True)
            _feature_matrix.index.set_names([id_name, 'time'], inplace=True)
            _feature_matrix.sort_index(level=1, kind='mergesort', inplace=True)
        else:
            # all rows have same cutoff time. set time and add passed columns
            num_rows = _feature_matrix.shape[0]
            time_index = pd.DatetimeIndex([time_last] * num_rows, name='time')
            _feature_matrix.set_index(time_index, append=True, inplace=True)
            if len(pass_columns) > 0:
                pass_through = group[['instance_id', cutoff_df_time_var] + pass_columns]
                pass_through.rename(columns={'instance_id': id_name,
                                             cutoff_df_time_var: 'time'},
                                    inplace=True)
                pass_through.set_index([id_name, 'time'], inplace=True)
                for col in pass_columns:
                    _feature_matrix[col] = pass_through[col]
        feature_matrix.append(_feature_matrix)

    feature_matrix = pd.concat(feature_matrix)
    return feature_matrix


def bin_cutoff_times(cuttoff_time, bin_size):
    binned_cutoff_time = cuttoff_time.copy()
    if type(bin_size) == int:
        binned_cutoff_time['time'] = binned_cutoff_time['time'].apply(lambda x: x / bin_size * bin_size)
    else:
        bin_size = _check_timedelta(bin_size).get_pandas_timedelta()
        binned_cutoff_time['time'] = datetime_round(binned_cutoff_time['time'], bin_size)
    return binned_cutoff_time


def save_csv_decorator(save_progress=None):
    def inner_decorator(method):
        @wraps(method)
        def wrapped(*args, **kwargs):
            if save_progress is None:
                r = method(*args, **kwargs)
            else:
                time = args[0].to_pydatetime()
                file_name = 'ft_' + time.strftime("%Y_%m_%d_%I-%M-%S-%f") + '.csv'
                file_path = os.path.join(save_progress, file_name)
                temp_dir = os.path.join(save_progress, 'temp')
                if not os.path.exists(temp_dir):
                    os.makedirs(temp_dir)
                temp_file_path = os.path.join(temp_dir, file_name)
                r = method(*args, **kwargs)
                r.to_csv(temp_file_path)
                os.rename(temp_file_path, file_path)
            return r
        return wrapped
    return inner_decorator


def approximate_features(features, cutoff_time, window, entityset, backend,
                         training_window=None, verbose=None, profile=None):
    '''Given a list of features and cutoff_times to be passed to
    calculate_feature_matrix, calculates approximate values of some features
    to speed up calculations.  Cutoff times are sorted into
    window-sized buckets and the approximate feature values are only calculated
    at one cutoff time for each bucket.


    ..note:: this only approximates DirectFeatures of AggregationPrimitives, on
        the target entity. In future versions, it may also be possible to
        approximate these features on other top-level entities

    Args:
        features (list[:class:`.PrimitiveBase`]): if these features are dependent
            on aggregation features on the prediction, the approximate values
            for the aggregation feature will be calculated

        cutoff_time (pd.DataFrame): specifies what time to calculate
            the features for each instance at.  A DataFrame with
            'instance_id' and 'time' columns.

        window (Timedelta or str): frequency to group instances with similar
            cutoff times by for features with costly calculations. For example,
            if bucket is 24 hours, all instances with cutoff times on the same
            day will use the same calculation for expensive features.

        entityset (:class:`.EntitySet`): An already initialized entityset.

        training_window (dict[str-> :class:`Timedelta`] or :class:`Timedelta`, optional):
            Window or windows defining how much older than the cutoff time data
            can be to be included when calculating the feature.  To specify
            which entities to apply windows to, use a dictionary mapping entity
            id -> Timedelta. If None, all older data is used.

        verbose (bool, optional): Print progress info.

        profile (bool, optional): Enables profiling if True

        save_progress (str, optional): path to save intermediate computational results
    '''
    if verbose:
        logger.info("Approximating features...")

    approx_fms_by_entity = {}
    all_approx_feature_set = None
    target_entity = features[0].entity
    target_index_var = target_entity.index

    to_approximate, all_approx_feature_set = gather_approximate_features(features, backend)

    target_time_colname = 'target_time'
    cutoff_time[target_time_colname] = cutoff_time['time']
    target_instance_colname = target_index_var
    cutoff_time[target_instance_colname] = cutoff_time['instance_id']
    approx_cutoffs = bin_cutoff_times(cutoff_time.copy(), window)
    cutoff_df_time_var = 'time'
    cutoff_df_instance_var = 'instance_id'
    # should this order be by dependencies so that calculate_feature_matrix
    # doesn't skip approximating something?
    for approx_entity_id, approx_features in to_approximate.items():
        # Gather associated instance_ids from the approximate entity
        cutoffs_with_approx_e_ids = approx_cutoffs.copy()
        frames = entityset.get_pandas_data_slice([approx_entity_id, target_entity.id],
                                                 target_entity.id,
                                                 cutoffs_with_approx_e_ids[target_instance_colname])

        if frames is not None:
            rvar = entityset.gen_relationship_var(target_entity.id, approx_entity_id)
            parent_instance_frame = frames[approx_entity_id][target_entity.id]
            cutoffs_with_approx_e_ids[rvar] = \
                cutoffs_with_approx_e_ids.merge(parent_instance_frame[[target_index_var, rvar]],
                                                on=target_index_var, how='left')[rvar].values
            new_approx_entity_index_var = rvar

            # Select only columns we care about
            columns_we_want = [target_instance_colname,
                               new_approx_entity_index_var,
                               cutoff_df_time_var,
                               target_time_colname]

            cutoffs_with_approx_e_ids = cutoffs_with_approx_e_ids[columns_we_want]
            cutoffs_with_approx_e_ids = cutoffs_with_approx_e_ids.drop_duplicates()
            cutoffs_with_approx_e_ids.dropna(subset=[new_approx_entity_index_var],
                                             inplace=True)
        else:
            cutoffs_with_approx_e_ids = pd.DataFrame()

        if cutoffs_with_approx_e_ids.empty:
            approx_fms_by_entity = gen_empty_approx_features_df(approx_features)
            continue

        cutoffs_with_approx_e_ids.sort_values([cutoff_df_time_var,
                                               new_approx_entity_index_var], inplace=True)
        # CFM assumes specific column names for cutoff_time argument
        rename = {new_approx_entity_index_var: cutoff_df_instance_var}
        cutoff_time_to_pass = cutoffs_with_approx_e_ids.rename(columns=rename)
        cutoff_time_to_pass = cutoff_time_to_pass[[cutoff_df_instance_var, cutoff_df_time_var]]

        cutoff_time_to_pass.drop_duplicates(inplace=True)
        approx_fm = calculate_feature_matrix(approx_features,
                                             cutoff_time=cutoff_time_to_pass,
                                             training_window=training_window,
                                             approximate=None,
                                             cutoff_time_in_index=False,
                                             profile=profile)

        approx_fms_by_entity[approx_entity_id] = approx_fm

    # Include entity because we only want to ignore features that
    # are base_features/dependencies of the top level entity we're
    # approximating.
    # For instance, if target entity is sessions, and we're
    # approximating customers.COUNT(sessions.COUNT(log.value)),
    # we could also just want the feature COUNT(log.value)
    # defined on sessions
    # as a first class feature in the feature matrix.
    # Unless we signify to only ignore it as a dependency of
    # a feature defined on customers, we would ignore computing it
    # and pandas_backend would error
    return approx_fms_by_entity, all_approx_feature_set


def datetime_round(dt, freq, round_up=False):
    """
    Taken from comments on the Pandas source: https://github.com/pandas-dev/pandas/issues/4314

    round down Timestamp series to a specified freq
    """
    if round_up:
        round_f = np.ceil
    else:
        round_f = np.floor
    dt = pd.DatetimeIndex(dt)
    freq = to_offset(freq).delta.value
    return pd.DatetimeIndex(((round_f(dt.asi8 / freq)) * freq).astype(np.int64))


def gather_approximate_features(features, backend):
    approximate_by_entity = defaultdict(list)
    approximate_feature_set = set([])
    for feature in features:
        if backend.feature_tree.uses_full_entity(feature):
            continue
        if isinstance(feature, DirectFeature):
            base_feature = feature.base_features[0]
            while not isinstance(base_feature, AggregationPrimitive):
                if isinstance(base_feature, DirectFeature):
                    base_feature = base_feature.base_features[0]
                else:
                    break
            if isinstance(base_feature, AggregationPrimitive):
                approx_entity = base_feature.entity.id
                approximate_by_entity[approx_entity].append(base_feature)
                approximate_feature_set.add(base_feature.hash())
    return approximate_by_entity, approximate_feature_set


def gen_empty_approx_features_df(approx_features):
    approx_entity_id = approx_features[0].entity.id
    df = pd.DataFrame(columns=[f.get_name() for f in approx_features])
    df.index.name = approx_features[0].entity.index
    approx_fms_by_entity = {approx_entity_id: df}
    return approx_fms_by_entity
