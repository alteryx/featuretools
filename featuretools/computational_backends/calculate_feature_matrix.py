from __future__ import division

import gc
import logging
import os
import shutil
import time
from builtins import zip
from collections import defaultdict
from datetime import datetime

import cloudpickle
import numpy as np
import pandas as pd

from .pandas_backend import PandasBackend
from .utils import (
    bin_cutoff_times,
    calc_num_per_chunk,
    create_client_and_cluster,
    gather_approximate_features,
    gen_empty_approx_features_df,
    get_next_chunk,
    save_csv_decorator
)

from featuretools.feature_base import AggregationFeature, FeatureBase
from featuretools.utils.gen_utils import (
    get_relationship_variable_id,
    make_tqdm_iterator
)
from featuretools.utils.wrangle import _check_time_type
from featuretools.variable_types import (
    DatetimeTimeIndex,
    NumericTimeIndex,
    PandasTypes
)

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
        features (list[:class:`.FeatureBase`]): Feature definitions to be calculated.

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
            all([isinstance(feature, FeatureBase) for feature in features])), \
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

    cutoff_time = cutoff_time.reset_index(drop=True)
    # handle how columns are names in cutoff_time
    # maybe add _check_time_dtype helper function
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
    # Check that cutoff_time time type matches entityset time type
    if entityset.time_type == NumericTimeIndex:
        if cutoff_time['time'].dtype.name not in PandasTypes._pandas_numerics:
            raise TypeError("cutoff_time times must be numeric: try casting "
                            "via pd.to_numeric(cutoff_time['time'])")
    elif entityset.time_type == DatetimeTimeIndex:
        if cutoff_time['time'].dtype.name not in PandasTypes._pandas_datetimes:
            raise TypeError("cutoff_time times must be datetime type: try casting via pd.to_datetime(cutoff_time['time'])")
    assert (cutoff_time[['instance_id', 'time']].duplicated().sum() == 0), \
        "Duplicated rows in cutoff time dataframe."
    pass_columns = [column_name for column_name in cutoff_time.columns[2:]]

    if _check_time_type(cutoff_time['time'].iloc[0]) is None:
        raise ValueError("cutoff_time time values must be datetime or numeric")

    backend = PandasBackend(entityset, features)

    # make sure dtype of instance_id in cutoff time
    # is same as column it references
    target_entity = features[0].entity
    dtype = entityset[target_entity.id].df[target_entity.index].dtype
    cutoff_time["instance_id"] = cutoff_time["instance_id"].astype(dtype)

    # Get dictionary of features to approximate
    if approximate is not None:
        to_approximate, all_approx_feature_set = gather_approximate_features(features, backend)
    else:
        to_approximate = defaultdict(list)
        all_approx_feature_set = None

    # Check if there are any non-approximated aggregation features
    no_unapproximated_aggs = True
    for feature in features:
        if isinstance(feature, AggregationFeature):
            # do not need to check if feature is in to_approximate since
            # only base features of direct features can be in to_approximate
            no_unapproximated_aggs = False
            break

        deps = feature.get_dependencies(deep=True, ignored=all_approx_feature_set)
        for dependency in deps:
            if (isinstance(dependency, AggregationFeature) and
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


def approximate_features(features, cutoff_time, window, entityset, backend,
                         training_window=None, profile=None):
    '''Given a list of features and cutoff_times to be passed to
    calculate_feature_matrix, calculates approximate values of some features
    to speed up calculations.  Cutoff times are sorted into
    window-sized buckets and the approximate feature values are only calculated
    at one cutoff time for each bucket.


    ..note:: this only approximates DirectFeatures of AggregationFeatures, on
        the target entity. In future versions, it may also be possible to
        approximate these features on other top-level entities

    Args:
        features (list[:class:`.FeatureBase`]): if these features are dependent
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
            path = entityset.find_path(approx_entity_id, target_entity.id)
            rvar = get_relationship_variable_id(path)
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
    from distributed import as_completed
    from dask.base import tokenize

    client = None
    cluster = None
    try:
        client, cluster = create_client_and_cluster(n_jobs=n_jobs,
                                                    num_tasks=len(chunks),
                                                    dask_kwargs=dask_kwargs,
                                                    entityset_size=entityset.__sizeof__())
        # scatter the entityset
        # denote future with leading underscore
        if verbose:
            start = time.time()
        es_token = "EntitySet-{}".format(tokenize(entityset))
        if es_token in client.list_datasets():
            if verbose:
                msg = "Using EntitySet persisted on the cluster as dataset {}"
                print(msg.format(es_token))
            _es = client.get_dataset(es_token)
        else:
            _es = client.scatter([entityset])[0]
            client.publish_dataset(**{_es.key: _es})

        # save features to a tempfile and scatter it
        pickled_feats = cloudpickle.dumps(features)
        _saved_features = client.scatter(pickled_feats)
        client.replicate([_es, _saved_features])
        if verbose:
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
