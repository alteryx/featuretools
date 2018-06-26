import os
import warnings
from collections import defaultdict
from functools import wraps

import numpy as np
import pandas as pd
import psutil
from pandas.tseries.frequencies import to_offset

from featuretools.primitives import (
    AggregationPrimitive,
    DirectFeature,
)
from featuretools.utils.wrangle import _check_timedelta


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


def n_jobs_to_workers(n_jobs):
    try:
        cpus = len(psutil.Process().cpu_affinity())
    except AttributeError:
        cpus = psutil.cpu_count()

    if n_jobs < 0:
        workers = max(cpus + 1 + n_jobs, 1)
    else:
        workers = min(n_jobs, cpus)

    assert workers > 0, "Need at least one worker"
    return workers


def create_client_and_cluster(n_jobs, num_tasks, dask_kwargs):
    from distributed import Client, LocalCluster
    cluster = None
    if 'cluster' in dask_kwargs:
        cluster = dask_kwargs['cluster']
    else:
        diagnostics_port = None
        if 'diagnostics_port' in dask_kwargs:
            diagnostics_port = dask_kwargs['diagnostics_port']
            del dask_kwargs['diagnostics_port']

        workers = n_jobs_to_workers(n_jobs)
        workers = min(workers, len(num_tasks))
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
    return client, cluster
