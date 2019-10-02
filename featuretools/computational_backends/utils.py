import logging
import os
import warnings
from functools import wraps

import pandas as pd
import psutil

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
        bin_size = _check_timedelta(bin_size)
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


def datetime_round(dt, freq):
    """
    round down Timestamp series to a specified freq
    """
    if not freq.is_absolute():
        raise ValueError("Unit is relative")

    # TODO: multitemporal units
    all_units = list(freq.times.keys())
    if len(all_units) == 1:
        unit = all_units[0]
        value = freq.times[unit]
        if unit == 'm':
            unit = 't'
        # No support for weeks in datetime.datetime
        if unit == 'w':
            unit = 'd'
            value = value * 7
        freq = str(value) + unit
        return dt.dt.floor(freq)
    else:
        assert "Frequency cannot have multiple temporal parameters"


def gather_approximate_features(feature_set):
    """
    Find features which can be approximated. Returned as a trie where the values
    are sets of feature names.

    Args:
        feature_set (FeatureSet): Features to search the dependencies of for
            features to approximate.

    Returns:
        Trie[RelationshipPath, set[str]]
    """
    approximate_feature_trie = Trie(default=set, path_constructor=RelationshipPath)

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
                node_feature_set = approximate_feature_trie.get_node(path).value
                node_feature_set.add(base_feature.unique_name())

    return approximate_feature_trie


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
    Client, LocalCluster = get_client_cluster()

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


def get_client_cluster():
    """
    Separated out the imports to make it easier to mock during testing
    """
    from distributed import Client, LocalCluster
    return Client, LocalCluster
