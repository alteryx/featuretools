from __future__ import division

import gc
import logging
import os
import shutil
import time
import warnings
from builtins import zip
from collections import defaultdict
from datetime import datetime
from functools import wraps
from sys import version_info

import cloudpickle
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
from featuretools.utils.wrangle import _check_time_type, _check_timedelta
from featuretools.variable_types import DatetimeTimeIndex, NumericTimeIndex

logger = logging.getLogger('featuretools.computational_backend')


def calculate_feature_matrix(features, entityset=None, cutoff_time=None, instance_ids=None,
                             entities=None, relationships=None,
                             cutoff_time_in_index=False,
                             training_window=None, approximate=None,
                             save_progress=None, verbose=False,
                             chunk_size=None, n_jobs=1, dask_kwargs=None,
                             profile=False):
    """Calculates a matrix for a given set of instance ids and calculation times.

    Args:
        features (list[PrimitiveBase]): Feature definitions to be calculated.

        entityset (EntitySet): An already initialized entityset. Required if `entities` and `relationships`
            not provided

        cutoff_time (pd.DataFrame or Datetime): Specifies at which time to calculate
            the features for each instance.  Can either be a DataFrame with
            'instance_id' and 'time' columns, DataFrame with the name of the
            index variable in the target entity and a time column, or a single
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

        cutoff_time_in_index (bool): If True, return a DataFrame with a MultiIndex
            where the second index is the cutoff time (first is instance id).
            DataFrame will be sorted by (time, instance_id).

        training_window (Timedelta, optional):
            Window defining how much older than the cutoff time data
            can be to be included when calculating the feature. If None, all older data is used.

        approximate (Timedelta or str): Frequency to group instances with similar
            cutoff times by for features with costly calculations. For example,
            if bucket is 24 hours, all instances with cutoff times on the same
            day will use the same calculation for expensive features.

        verbose (bool, optional): Print progress info. The time granularity is
            per chunk.

        profile (bool, optional): Enables profiling if True.

        chunk_size (int or float or None or "cutoff time"): Number of rows of
            output feature matrix to calculate at time. If passed an integer
            greater than 0, will try to use that many rows per chunk. If passed
            a float value between 0 and 1 sets the chunk size to that
            percentage of all instances. If passed the string "cutoff time",
            rows are split per cutoff time.

        n_jobs (int, optional): number of parallel processes to use when
            calculating feature matrix

        dask_kwargs (dict, optional): Dictionary of keyword arguments to be
            passed when creating the dask client and scheduler. Even if n_jobs
            is not set, using `dask_kwargs` will enable multiprocessing.
            Main parameters:

            cluster (str or dask.distributed.LocalCluster):
                cluster or address of cluster to send tasks to. If unspecified,
                a cluster will be created.
            diagnostics port (int):
                port number to use for web dashboard.  If left unspecified, web
                interface will not be enabled.

            Valid keyword arguments for LocalCluster will also be accepted.

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

    target_entity = entityset[features[0].entity.id]
    pass_columns = []

    if not isinstance(cutoff_time, pd.DataFrame):
        if isinstance(cutoff_time, list):
            raise TypeError("cutoff_time must be a single value or DataFrame")

        if cutoff_time is None:
            if entityset.time_type == NumericTimeIndex:
                cutoff_time = np.inf
            else:
                cutoff_time = datetime.now()

        if instance_ids is None:
            index_var = target_entity.index
            instance_ids = target_entity.df[index_var].tolist()

        cutoff_time = [cutoff_time] * len(instance_ids)
        map_args = [(id, time) for id, time in zip(instance_ids, cutoff_time)]
        cutoff_time = pd.DataFrame(map_args, columns=['instance_id', 'time'])
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
        if cutoff_time['time'].dtype == object:
            if (entityset.time_type == NumericTimeIndex and
                    cutoff_time['time'].dtype.name.find('int') == -1 and
                    cutoff_time['time'].dtype.name.find('float') == -1):
                raise TypeError("cutoff_time times must be numeric: try casting via pd.to_numeric(cutoff_time['time'])")
            elif (entityset.time_type == DatetimeTimeIndex and
                  cutoff_time['time'].dtype.name.find('time') == -1):
                raise TypeError("cutoff_time times must be datetime type: try casting via pd.to_datetime(cutoff_time['time'])")
        pass_columns = [column_name for column_name in cutoff_time.columns[2:]]

    if _check_time_type(cutoff_time['time'].iloc[0]) is None:
        raise ValueError("cutoff_time time values must be datetime or numeric")

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
    num_per_chunk = calc_num_per_chunk(chunk_size, cutoff_time.shape)

    if approximate is not None:
        # If there are approximated aggs, bin times
        binned_cutoff_time = bin_cutoff_times(cutoff_time.copy(), approximate)

        # Think about collisions: what if original time is a feature
        binned_cutoff_time[target_time] = cutoff_time[cutoff_df_time_var]

        cutoff_time_to_pass = binned_cutoff_time

    else:
        cutoff_time_to_pass = cutoff_time

    if num_per_chunk == "cutoff time":
        iterator = cutoff_time_to_pass.groupby(cutoff_df_time_var)
    else:
        iterator = get_next_chunk(cutoff_time=cutoff_time_to_pass,
                                  time_variable=cutoff_df_time_var,
                                  num_per_chunk=num_per_chunk)

    chunks = []
    if num_per_chunk == "cutoff time":
        for _, group in iterator:
            chunks.append(group)
    else:
        for chunk in iterator:
            chunks.append(chunk)

    if n_jobs != 1 or dask_kwargs is not None:
        feature_matrix = parallel_calculate_chunks(chunks=chunks,
                                                   features=features,
                                                   approximate=approximate,
                                                   training_window=training_window,
                                                   verbose=verbose,
                                                   save_progress=save_progress,
                                                   entityset=entityset,
                                                   n_jobs=n_jobs,
                                                   no_unapproximated_aggs=no_unapproximated_aggs,
                                                   cutoff_df_time_var=cutoff_df_time_var,
                                                   target_time=target_time,
                                                   pass_columns=pass_columns,
                                                   dask_kwargs=dask_kwargs or {})
    else:
        feature_matrix = linear_calculate_chunks(chunks=chunks,
                                                 features=features,
                                                 approximate=approximate,
                                                 training_window=training_window,
                                                 profile=profile,
                                                 verbose=verbose,
                                                 save_progress=save_progress,
                                                 entityset=entityset,
                                                 no_unapproximated_aggs=no_unapproximated_aggs,
                                                 cutoff_df_time_var=cutoff_df_time_var,
                                                 target_time=target_time,
                                                 pass_columns=pass_columns)

    feature_matrix = pd.concat(feature_matrix)

    feature_matrix.sort_index(level='time', kind='mergesort', inplace=True)
    if not cutoff_time_in_index:
        feature_matrix.reset_index(level='time', drop=True, inplace=True)

    if save_progress and os.path.exists(os.path.join(save_progress, 'temp')):
        shutil.rmtree(os.path.join(save_progress, 'temp'))

    return feature_matrix


def calculate_chunk(chunk, features, approximate, training_window,
                    profile, verbose, save_progress,
                    no_unapproximated_aggs, cutoff_df_time_var, target_time,
                    pass_columns, backend=None, entityset=None):
    if not isinstance(features, list):
        features = cloudpickle.loads(features)

    assert entityset is not None or backend is not None, "Must provide either"\
        " entityset or backend to calculate_chunk"
    if entityset is None:
        entityset = backend.entityset
    if backend is None:
        backend = PandasBackend(entityset, features)

    feature_matrix = []
    entityset = backend.entityset
    if no_unapproximated_aggs and approximate is not None:
        if entityset.time_type == NumericTimeIndex:
            chunk_time = np.inf
        else:
            chunk_time = datetime.now()

    for _, group in chunk.groupby(cutoff_df_time_var):
        # if approximating, calculate the approximate features
        if approximate is not None:
            precalculated_features, all_approx_feature_set = approximate_features(
                features,
                group,
                window=approximate,
                entityset=backend.entityset,
                backend=backend,
                training_window=training_window,
                profile=profile
            )
        else:
            precalculated_features = None
            all_approx_feature_set = None

        @save_csv_decorator(save_progress)
        def calc_results(time_last, ids, precalculated_features=None, training_window=None):
            matrix = backend.calculate_all_features(ids, time_last,
                                                    training_window=training_window,
                                                    precalculated_features=precalculated_features,
                                                    ignored=all_approx_feature_set,
                                                    profile=profile)
            return matrix

        # if all aggregations have been approximated, can calculate all together
        if no_unapproximated_aggs and approximate is not None:
            grouped = [[chunk_time, group]]
        else:
            # if approximated features, set cutoff_time to unbinned time
            if precalculated_features is not None:
                group[cutoff_df_time_var] = group[target_time]

            grouped = group.groupby(cutoff_df_time_var, sort=True)

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
                time_index = pd.Index([time_last] * num_rows, name='time')
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
                         training_window=None, profile=None):
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

        training_window (`Timedelta`, optional):
            Window defining how much older than the cutoff time data
            can be to be included when calculating the feature. If None, all older data is used.

        profile (bool, optional): Enables profiling if True

        save_progress (str, optional): path to save intermediate computational results
    '''
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
                cutoffs_with_approx_e_ids.merge(parent_instance_frame[[rvar]],
                                                left_on=target_index_var,
                                                right_index=True,
                                                how='left')[rvar].values
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
                                             entityset,
                                             cutoff_time=cutoff_time_to_pass,
                                             training_window=training_window,
                                             approximate=None,
                                             cutoff_time_in_index=False,
                                             chunk_size=cutoff_time_to_pass.shape[0],
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


def calc_num_per_chunk(chunk_size, shape):
    """
    Given a chunk size and the shape of the feature matrix to split into
    chunk, returns the number of rows there should be per chunk
    """
    if isinstance(chunk_size, float) and chunk_size > 0 and chunk_size < 1:
        num_per_chunk = int(shape[0] * float(chunk_size))
        # must be at least 1 cutoff per chunk
        num_per_chunk = max(1, num_per_chunk)
    elif isinstance(chunk_size, int) and chunk_size >= 1:
        if chunk_size > shape[0]:
            warnings.warn("Chunk size is greater than size of feature matrix")
            num_per_chunk = shape[0]
        else:
            num_per_chunk = chunk_size
    elif chunk_size is None:
        num_per_chunk = max(int(shape[0] * .1), 10)
    elif chunk_size == "cutoff time":
        num_per_chunk = "cutoff time"
    else:
        raise ValueError("chunk_size must be None, a float between 0 and 1,"
                         "a positive integer, or the string 'cutoff time'")
    return num_per_chunk


def get_next_chunk(cutoff_time, time_variable, num_per_chunk):
    """
    Generator function that takes a DataFrame of cutoff times and the number of
    rows to include per chunk and returns an iterator of the resulting chunks.

    Args:
        cutoff_time (pd.DataFrame): dataframe of cutoff times to chunk
        time_variable (str): name of time column in cutoff_time dataframe
        num_per_chunk (int): maximum number of rows to include in a chunk
    """
    # if chunk_size is 100%, return DataFrame immediately and stop iteration
    if cutoff_time.shape[0] <= num_per_chunk:
        yield cutoff_time
        return

    # split rows of cutoff_time into groups based on time variable
    grouped = cutoff_time.groupby(time_variable, sort=False)

    # sort groups by size, largest first
    groups = grouped.size().sort_values(ascending=False).index

    # list of partially filled chunks
    chunks = []

    # iterate through each group and try to make completely filled chunks
    for group_name in groups:
        # get locations in cutoff_time (iloc) of all rows in group
        group = grouped.groups[group_name].values.tolist()

        # divide up group into slices if too large to fit in a single chunk
        group_slices = []
        if len(group) > num_per_chunk:
            for i in range(0, len(group), num_per_chunk):
                    group_slices.append(group[i: i + num_per_chunk])
        else:
            group_slices.append(group)

        # for each slice of the group, try to find a chunk it can fit in
        for group_slice in group_slices:
            # if slice is exactly the number of rows for a chunk, yield the
            # slice's rows of cutoff_time as the next chunk and move on
            if len(group_slice) == num_per_chunk:
                yield cutoff_time.loc[group_slice]
                continue

            # if not, look for partially filled chunks that have room
            found_chunk = False
            for i in range(len(chunks)):
                chunk = chunks[i]
                if len(chunk) + len(group_slice) <= num_per_chunk:
                    chunk.extend(group_slice)
                    found_chunk = True
                    if len(chunk) == num_per_chunk:
                        # if chunk is full, pop from partial list and yield
                        loc_list = chunks.pop(i)
                        yield cutoff_time.loc[loc_list]
                    break

            # if no chunk has room, this slice becomes another partial chunk
            if not found_chunk:
                chunks.append(group_slice)

    # after iterating through every group, yield any remaining partial chunks
    for chunk in chunks:
        yield cutoff_time.loc[chunk]


def linear_calculate_chunks(chunks, features, approximate, training_window,
                            profile, verbose, save_progress, entityset,
                            no_unapproximated_aggs, cutoff_df_time_var,
                            target_time, pass_columns):
    backend = PandasBackend(entityset, features)
    feature_matrix = []

    # if verbose, create progess bar
    if verbose:
        pbar_string = ("Elapsed: {elapsed} | Remaining: {remaining} | "
                       "Progress: {l_bar}{bar}| "
                       "Calculated: {n}/{total} chunks")
        chunks = make_tqdm_iterator(iterable=chunks,
                                    total=len(chunks),
                                    bar_format=pbar_string)

    for chunk in chunks:
        _feature_matrix = calculate_chunk(chunk, features, approximate,
                                          training_window,
                                          profile, verbose,
                                          save_progress,
                                          no_unapproximated_aggs,
                                          cutoff_df_time_var,
                                          target_time, pass_columns,
                                          backend=backend)
        feature_matrix.append(_feature_matrix)
        # Do a manual garbage collection in case objects from calculate_chunk
        # weren't collected automatically
        gc.collect()
    if verbose:
        chunks.close()
    return feature_matrix


def parallel_calculate_chunks(chunks, features, approximate, training_window,
                              verbose, save_progress, entityset, n_jobs,
                              no_unapproximated_aggs, cutoff_df_time_var,
                              target_time, pass_columns, dask_kwargs=None):
    from distributed import Client, LocalCluster, as_completed
    from dask.base import tokenize

    client = None
    cluster = None
    try:
        if 'cluster' in dask_kwargs:
            cluster = dask_kwargs['cluster']
        else:
            diagnostics_port = None
            if 'diagnostics_port' in dask_kwargs:
                diagnostics_port = dask_kwargs['diagnostics_port']
                del dask_kwargs['diagnostics_port']

            workers = n_jobs_to_workers(n_jobs)
            workers = min(workers, len(chunks))
            cluster = LocalCluster(n_workers=workers,
                                   threads_per_worker=1,
                                   diagnostics_port=diagnostics_port,
                                   **dask_kwargs)
            # if cluster has bokeh port, notify user if unxepected port number
            if diagnostics_port is not None:
                if hasattr(cluster, 'scheduler') and cluster.scheduler:
                    info = cluster.scheduler.identity()
                    if 'bokeh' in info['services']:
                        msg = "Dashboard started on port {}"
                        print(msg.format(info['services']['bokeh']))

        client = Client(cluster)
        # scatter the entityset
        # denote future with leading underscore
        start = time.time()
        es_token = "EntitySet-{}".format(tokenize(entityset))
        if es_token in client.list_datasets():
            print("Using EntitySet persisted on the cluster as dataset %s" % (es_token))
            _es = client.get_dataset(es_token)
        else:
            _es = client.scatter([entityset])[0]
            client.publish_dataset(**{_es.key: _es})

        # save features to a tempfile and scatter it
        pickled_feats = cloudpickle.dumps(features)
        _saved_features = client.scatter(pickled_feats)
        client.replicate([_es, _saved_features])
        end = time.time()
        scatter_time = end - start
        scatter_string = "EntitySet scattered to workers in {:.3f} seconds"
        print(scatter_string.format(scatter_time))

        # map chunks
        # TODO: consider handling task submission dask kwargs
        _chunks = client.map(calculate_chunk,
                             chunks,
                             features=_saved_features,
                             entityset=_es,
                             approximate=approximate,
                             training_window=training_window,
                             profile=False,
                             verbose=False,
                             save_progress=save_progress,
                             no_unapproximated_aggs=no_unapproximated_aggs,
                             cutoff_df_time_var=cutoff_df_time_var,
                             target_time=target_time,
                             pass_columns=pass_columns)

        feature_matrix = []
        iterator = as_completed(_chunks).batches()
        if verbose:
            pbar_str = ("Elapsed: {elapsed} | Remaining: {remaining} | "
                        "Progress: {l_bar}{bar}| "
                        "Calculated: {n}/{total} chunks")
            pbar = make_tqdm_iterator(total=len(_chunks), bar_format=pbar_str)
        for batch in iterator:
            results = client.gather(batch)
            for result in results:
                feature_matrix.append(result)
                if verbose:
                    pbar.update()
        if verbose:
            pbar.close()
    except Exception:
        raise
    finally:
        if 'cluster' not in dask_kwargs and cluster is not None:
            cluster.close()
        if client is not None:
            client.close()

    return feature_matrix


def n_jobs_to_workers(n_jobs):
    if version_info.major == 2:
        import multiprocessing
        cpus = multiprocessing.cpu_count()
    else:
        cpus = len(os.sched_getaffinity(0))

    if n_jobs < 0:
        workers = max(cpus + 1 + n_jobs, 1)
    else:
        workers = min(n_jobs, cpus)

    assert workers > 0, "Need at least one worker"
    return workers
