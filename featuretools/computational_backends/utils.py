import logging
import os
import typing
import warnings
from datetime import datetime
from functools import wraps

import numpy as np
import pandas as pd
import psutil
from woodwork.logical_types import Datetime, Double

from featuretools.entityset.relationship import RelationshipPath
from featuretools.feature_base import AggregationFeature, DirectFeature
from featuretools.utils import Trie
from featuretools.utils.gen_utils import Library, import_or_none, is_instance
from featuretools.utils.wrangle import _check_time_type, _check_timedelta

dd = import_or_none("dask.dataframe")
logger = logging.getLogger("featuretools.computational_backend")


def bin_cutoff_times(cutoff_time, bin_size):
    binned_cutoff_time = cutoff_time.ww.copy()
    if isinstance(bin_size, int):
        binned_cutoff_time["time"] = binned_cutoff_time["time"].apply(
            lambda x: x / bin_size * bin_size,
        )
    else:
        bin_size = _check_timedelta(bin_size)
        binned_cutoff_time["time"] = datetime_round(
            binned_cutoff_time["time"],
            bin_size,
        )
    return binned_cutoff_time


def save_csv_decorator(save_progress=None):
    def inner_decorator(method):
        @wraps(method)
        def wrapped(*args, **kwargs):
            if save_progress is None:
                r = method(*args, **kwargs)
            else:
                time = args[0].to_pydatetime()
                file_name = "ft_" + time.strftime("%Y_%m_%d_%I-%M-%S-%f") + ".csv"
                file_path = os.path.join(save_progress, file_name)
                temp_dir = os.path.join(save_progress, "temp")
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
        if unit == "m":
            unit = "t"
        # No support for weeks in datetime.datetime
        if unit == "w":
            unit = "d"
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
        if feature_set.uses_full_dataframe(feature, check_dependents=True):
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
    df.index.name = approx_features[0].dataframe.ww.index
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
    if "cluster" in dask_kwargs:
        cluster = dask_kwargs["cluster"]
    else:
        # diagnostics_port sets the default port to launch bokeh web interface
        # if it is set to None web interface will not be launched
        diagnostics_port = None
        if "diagnostics_port" in dask_kwargs:
            diagnostics_port = dask_kwargs["diagnostics_port"]
            del dask_kwargs["diagnostics_port"]

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
        if "memory_limit" in dask_kwargs:
            memory_limit = dask_kwargs["memory_limit"]
            del dask_kwargs["memory_limit"]
        else:
            total_memory = psutil.virtual_memory().total
            memory_limit = int(total_memory / float(workers))

        cluster = LocalCluster(
            n_workers=workers,
            threads_per_worker=1,
            diagnostics_port=diagnostics_port,
            memory_limit=memory_limit,
            **dask_kwargs,
        )

        # if cluster has bokeh port, notify user if unexpected port number
        if diagnostics_port is not None:
            if hasattr(cluster, "scheduler") and cluster.scheduler:
                info = cluster.scheduler.identity()
                if "bokeh" in info["services"]:
                    msg = "Dashboard started on port {}"
                    print(msg.format(info["services"]["bokeh"]))

    client = Client(cluster)

    warned_of_memory = False
    for worker in list(client.scheduler_info()["workers"].values()):
        worker_limit = worker["memory_limit"]
        if worker_limit < entityset_size:
            raise ValueError("Insufficient memory to use this many workers")
        elif worker_limit < 2 * entityset_size and not warned_of_memory:
            logger.warning(
                "Worker memory is between 1 to 2 times the memory"
                " size of the EntitySet. If errors occur that do"
                " not occur with n_jobs equals 1, this may be the "
                "cause.  See https://featuretools.alteryx.com/en/stable/guides/performance.html#parallel-feature-computation"
                " for more information.",
            )
            warned_of_memory = True

    return client, cluster


def get_client_cluster():
    """
    Separated out the imports to make it easier to mock during testing
    """
    distributed = import_or_none("distributed")
    Client = distributed.Client
    LocalCluster = distributed.LocalCluster

    return Client, LocalCluster


if dd:
    CutoffTimeType = typing.Union[dd.DataFrame, pd.DataFrame, str, datetime]
else:
    CutoffTimeType = typing.Union[pd.DataFrame, str, datetime]


def _validate_cutoff_time(
    cutoff_time: CutoffTimeType,
    target_dataframe,
):
    """
    Verify that the cutoff time is a single value or a pandas dataframe with the proper columns
    containing no duplicate rows
    """
    if is_instance(cutoff_time, dd, "DataFrame"):
        msg = (
            "cutoff_time should be a Pandas DataFrame: "
            "computing cutoff_time, this may take a while"
        )
        warnings.warn(msg)
        cutoff_time = cutoff_time.compute()

    if isinstance(cutoff_time, pd.DataFrame):
        cutoff_time = cutoff_time.reset_index(drop=True)

        if "instance_id" not in cutoff_time.columns:
            if target_dataframe.ww.index not in cutoff_time.columns:
                raise AttributeError(
                    "Cutoff time DataFrame must contain a column with either the same name"
                    ' as the target dataframe index or a column named "instance_id"',
                )
            # rename to instance_id
            cutoff_time.rename(
                columns={target_dataframe.ww.index: "instance_id"},
                inplace=True,
            )

        if "time" not in cutoff_time.columns:
            if (
                target_dataframe.ww.time_index
                and target_dataframe.ww.time_index not in cutoff_time.columns
            ):
                raise AttributeError(
                    "Cutoff time DataFrame must contain a column with either the same name"
                    ' as the target dataframe time_index or a column named "time"',
                )
            # rename to time
            cutoff_time.rename(
                columns={target_dataframe.ww.time_index: "time"},
                inplace=True,
            )

        # Make sure user supplies only one valid name for instance id and time columns
        if (
            "instance_id" in cutoff_time.columns
            and target_dataframe.ww.index in cutoff_time.columns
            and "instance_id" != target_dataframe.ww.index
        ):
            raise AttributeError(
                'Cutoff time DataFrame cannot contain both a column named "instance_id" and a column'
                " with the same name as the target dataframe index",
            )
        if (
            "time" in cutoff_time.columns
            and target_dataframe.ww.time_index in cutoff_time.columns
            and "time" != target_dataframe.ww.time_index
        ):
            raise AttributeError(
                'Cutoff time DataFrame cannot contain both a column named "time" and a column'
                " with the same name as the target dataframe time index",
            )

        assert (
            cutoff_time[["instance_id", "time"]].duplicated().sum() == 0
        ), "Duplicated rows in cutoff time dataframe."
    if isinstance(cutoff_time, str):
        try:
            cutoff_time = pd.to_datetime(cutoff_time)
        except ValueError as e:
            raise ValueError(f"While parsing cutoff_time: {str(e)}")
        except OverflowError as e:
            raise OverflowError(f"While parsing cutoff_time: {str(e)}")
    else:
        if isinstance(cutoff_time, list):
            raise TypeError("cutoff_time must be a single value or DataFrame")

    return cutoff_time


def _check_cutoff_time_type(cutoff_time, es_time_type):
    """
    Check that the cutoff time values are of the proper type given the entityset time type
    """
    # Check that cutoff_time time type matches entityset time type
    if isinstance(cutoff_time, tuple):
        cutoff_time_value = cutoff_time[0]
        time_type = _check_time_type(cutoff_time_value)
        is_numeric = time_type == "numeric"
        is_datetime = time_type == Datetime
    else:
        cutoff_time_col = cutoff_time.ww["time"]
        is_numeric = cutoff_time_col.ww.schema.is_numeric
        is_datetime = cutoff_time_col.ww.schema.is_datetime

    if es_time_type == "numeric" and not is_numeric:
        raise TypeError(
            "cutoff_time times must be numeric: try casting " "via pd.to_numeric()",
        )
    if es_time_type == Datetime and not is_datetime:
        raise TypeError(
            "cutoff_time times must be datetime type: try casting "
            "via pd.to_datetime()",
        )


def replace_inf_values(feature_matrix, replacement_value=np.nan, columns=None):
    """Replace all ``np.inf`` values in a feature matrix with the specified replacement value.

    Args:
        feature_matrix (DataFrame): DataFrame whose columns are feature names and rows are instances
        replacement_value (int, float, str, optional): Value with which ``np.inf`` values will be replaced
        columns (list[str], optional): A list specifying which columns should have values replaced. If None,
            values will be replaced for all columns.

    Returns:
        feature_matrix

    """
    if columns is None:
        feature_matrix = feature_matrix.replace([np.inf, -np.inf], replacement_value)
    else:
        feature_matrix[columns] = feature_matrix[columns].replace(
            [np.inf, -np.inf],
            replacement_value,
        )
    return feature_matrix


def get_ww_types_from_features(
    features,
    entityset,
    pass_columns=None,
    cutoff_time=None,
):
    """Given a list of features and entityset (and optionally a list of pass
    through columns and the cutoff time dataframe), returns the logical types,
    semantic tags,and origin of each column in the feature matrix.  Both
    pass_columns and cutoff_time will need to be supplied in order to get the
    type information for the pass through columns
    """
    if pass_columns is None:
        pass_columns = []
    logical_types = {}
    semantic_tags = {}
    origins = {}

    for feature in features:
        names = feature.get_feature_names()
        for name in names:
            logical_types[name] = feature.column_schema.logical_type
            semantic_tags[name] = feature.column_schema.semantic_tags.copy()
            semantic_tags[name] -= {"index", "time_index"}

            if logical_types[name] is None and "numeric" in semantic_tags[name]:
                logical_types[name] = Double
            if all([f.primitive is None for f in feature.get_dependencies(deep=True)]):
                origins[name] = "base"
            else:
                origins[name] = "engineered"

    if pass_columns:
        cutoff_schema = cutoff_time.ww.schema
        for column in pass_columns:
            logical_types[column] = cutoff_schema.logical_types[column]
            semantic_tags[column] = cutoff_schema.semantic_tags[column]
            origins[column] = "base"

    if entityset.dataframe_type in (Library.DASK, Library.SPARK):
        target_dataframe_name = features[0].dataframe_name
        table_schema = entityset[target_dataframe_name].ww.schema
        index_col = table_schema.index
        logical_types[index_col] = table_schema.logical_types[index_col]
        semantic_tags[index_col] = table_schema.semantic_tags[index_col]
        semantic_tags[index_col] -= {"index"}
        origins[index_col] = "base"

    ww_init = {
        "logical_types": logical_types,
        "semantic_tags": semantic_tags,
        "column_origins": origins,
    }
    return ww_init
