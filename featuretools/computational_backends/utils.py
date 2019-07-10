# -*- coding: utf-8 -*-

import logging
import os
import warnings
from functools import wraps

import numpy as np
import pandas as pd
import psutil
from distributed import Client, LocalCluster
from pandas.tseries.frequencies import to_offset

from featuretools.entityset.relationship import RelationshipPath
from featuretools.feature_base import AggregationFeature, DirectFeature
from featuretools.utils import Trie
from featuretools.utils.wrangle import _check_timedelta

logger = logging.getLogger('featuretools.computational_backend')


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


def gather_approximate_features(feature_set):
    # A trie where the edges are RelationshipPaths and the nodes contain lists
    # of features.
    approximate_feature_trie = Trie(default=list, path_constructor=RelationshipPath)

    # A set of feature names.
    approximate_feature_set = set()

    for feature in feature_set.target_features:
        if feature_set.uses_full_entity(feature, check_dependents=True):
            continue

        if isinstance(feature, DirectFeature):
            path = feature.relationship_path
            base_feature = feature.base_features[0]

            while isinstance(base_feature, DirectFeature):
                path = path + base_feature.relationship_path
                base_feature = base_feature.base_features[0]

            if isinstance(base_feature, AggregationFeature):
                feature_list = approximate_feature_trie.get_node(path).value
                feature_list.append(base_feature)
                approximate_feature_set.add(base_feature.unique_name())

    return approximate_feature_trie, approximate_feature_set


def gen_empty_approx_features_df(approx_features):
    df = pd.DataFrame(columns=[f.get_name() for f in approx_features])
    df.index.name = approx_features[0].entity.index
    return df


def n_jobs_to_workers(n_jobs):
    try:
        cpus = len(psutil.Process().cpu_affinity())
    except AttributeError:
        cpus = psutil.cpu_count()

    # Taken from sklearn parallel_backends code
    # https://github.com/scikit-learn/scikit-learn/blob/27bbdb570bac062c71b3bb21b0876fd78adc9f7e/sklearn/externals/joblib/_parallel_backends.py#L120
    if n_jobs < 0:
        workers = max(cpus + 1 + n_jobs, 1)
    else:
        workers = min(n_jobs, cpus)

    assert workers > 0, "Need at least one worker"
    return workers


def create_client_and_cluster(n_jobs, dask_kwargs, entityset_size):
    cluster = None
    if 'cluster' in dask_kwargs:
        cluster = dask_kwargs['cluster']
    else:
        # diagnostics_port sets the default port to launch bokeh web interface
        # if it is set to None web interface will not be launched
        diagnostics_port = None
        if 'diagnostics_port' in dask_kwargs:
            diagnostics_port = dask_kwargs['diagnostics_port']
            del dask_kwargs['diagnostics_port']

        workers = n_jobs_to_workers(n_jobs)
        if n_jobs != -1 and workers < n_jobs:
            warning_string = "{} workers requested, but only {} workers created."
            warning_string = warning_string.format(n_jobs, workers)
            warnings.warn(warning_string)

        # Distributed default memory_limit for worker is 'auto'. It calculates worker
        # memory limit as total virtual memory divided by the number
        # of cores available to the workers (alwasy 1 for featuretools setup).
        # This means reducing the number of workers does not increase the memory
        # limit for other workers.  Featuretools default is to calculate memory limit
        # as total virtual memory divided by number of workers. To use distributed
        # default memory limit, set dask_kwargs['memory_limit']='auto'
        if 'memory_limit' in dask_kwargs:
            memory_limit = dask_kwargs['memory_limit']
            del dask_kwargs['memory_limit']
        else:
            total_memory = psutil.virtual_memory().total
            memory_limit = int(total_memory / float(workers))

        cluster = LocalCluster(n_workers=workers,
                               threads_per_worker=1,
                               diagnostics_port=diagnostics_port,
                               memory_limit=memory_limit,
                               **dask_kwargs)

        # if cluster has bokeh port, notify user if unexpected port number
        if diagnostics_port is not None:
            if hasattr(cluster, 'scheduler') and cluster.scheduler:
                info = cluster.scheduler.identity()
                if 'bokeh' in info['services']:
                    msg = "Dashboard started on port {}"
                    print(msg.format(info['services']['bokeh']))

    client = Client(cluster)

    warned_of_memory = False
    for worker in list(client.scheduler_info()['workers'].values()):
        worker_limit = worker['memory_limit']
        if worker_limit < entityset_size:
            raise ValueError("Insufficient memory to use this many workers")
        elif worker_limit < 2 * entityset_size and not warned_of_memory:
            logger.warn("Worker memory is between 1 to 2 times the memory"
                        " size of the EntitySet. If errors occur that do"
                        " not occur with n_jobs equals 1, this may be the "
                        "cause.  See https://docs.featuretools.com/guides/parallel.html"
                        " for more information.")
            warned_of_memory = True

    return client, cluster
