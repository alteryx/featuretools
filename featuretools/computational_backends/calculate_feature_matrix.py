import logging
import math
import os
import shutil
import time
import warnings
from datetime import datetime

import cloudpickle
import numpy as np
import pandas as pd
from woodwork.logical_types import (
    Age,
    AgeNullable,
    Boolean,
    BooleanNullable,
    Integer,
    IntegerNullable,
)

from featuretools.computational_backends.feature_set import FeatureSet
from featuretools.computational_backends.feature_set_calculator import (
    FeatureSetCalculator,
)
from featuretools.computational_backends.utils import (
    _check_cutoff_time_type,
    _validate_cutoff_time,
    bin_cutoff_times,
    create_client_and_cluster,
    gather_approximate_features,
    gen_empty_approx_features_df,
    get_ww_types_from_features,
    save_csv_decorator,
)
from featuretools.entityset.relationship import RelationshipPath
from featuretools.feature_base import AggregationFeature, FeatureBase
from featuretools.utils import Trie
from featuretools.utils.gen_utils import (
    Library,
    import_or_none,
    import_or_raise,
    is_instance,
    make_tqdm_iterator,
)

dd = import_or_none("dask.dataframe")
ps = import_or_none("pyspark.pandas")

logger = logging.getLogger("featuretools.computational_backend")

PBAR_FORMAT = "Elapsed: {elapsed} | Progress: {l_bar}{bar}"
FEATURE_CALCULATION_PERCENTAGE = (
    0.95  # make total 5% higher to allot time for wrapping up at end
)


def calculate_feature_matrix(
    features,
    entityset=None,
    cutoff_time=None,
    instance_ids=None,
    dataframes=None,
    relationships=None,
    cutoff_time_in_index=False,
    training_window=None,
    approximate=None,
    save_progress=None,
    verbose=False,
    chunk_size=None,
    n_jobs=1,
    dask_kwargs=None,
    progress_callback=None,
    include_cutoff_time=True,
):
    """Calculates a matrix for a given set of instance ids and calculation times.

    Args:
        features (list[:class:`.FeatureBase`]): Feature definitions to be calculated.

        entityset (EntitySet): An already initialized entityset. Required if `dataframes` and `relationships`
            not provided

        cutoff_time (pd.DataFrame or Datetime): Specifies times at which to calculate
            the features for each instance. The resulting feature matrix will use data
            up to and including the cutoff_time. Can either be a DataFrame or a single
            value. If a DataFrame is passed the instance ids for which to calculate features
            must be in a column with the same name as the target dataframe index or a column
            named `instance_id`. The cutoff time values in the DataFrame must be in a column with
            the same name as the target dataframe time index or a column named `time`. If the
            DataFrame has more than two columns, any additional columns will be added to the
            resulting feature matrix. If a single value is passed, this value will be used for
            all instances.

        instance_ids (list): List of instances to calculate features on. Only
            used if cutoff_time is a single datetime.

        dataframes (dict[str -> tuple(DataFrame, str, str, dict[str -> str/Woodwork.LogicalType], dict[str->str/set], boolean)]):
            Dictionary of DataFrames. Entries take the format
            {dataframe name -> (dataframe, index column, time_index, logical_types, semantic_tags, make_index)}.
            Note that only the dataframe is required. If a Woodwork DataFrame is supplied, any other parameters
            will be ignored.

        relationships (list[(str, str, str, str)]): list of relationships
            between dataframes. List items are a tuple with the format
            (parent dataframe name, parent column, child dataframe name, child column).

        cutoff_time_in_index (bool): If True, return a DataFrame with a MultiIndex
            where the second index is the cutoff time (first is instance id).
            DataFrame will be sorted by (time, instance_id).

        training_window (Timedelta or str, optional):
            Window defining how much time before the cutoff time data
            can be used when calculating features. If ``None``, all data before cutoff time is used.
            Defaults to ``None``.

        approximate (Timedelta or str): Frequency to group instances with similar
            cutoff times by for features with costly calculations. For example,
            if bucket is 24 hours, all instances with cutoff times on the same
            day will use the same calculation for expensive features.

        verbose (bool, optional): Print progress info. The time granularity is
            per chunk.

        chunk_size (int or float or None): maximum number of rows of
            output feature matrix to calculate at time. If passed an integer
            greater than 0, will try to use that many rows per chunk. If passed
            a float value between 0 and 1 sets the chunk size to that
            percentage of all rows. if None, and n_jobs > 1 it will be set to 1/n_jobs

        n_jobs (int, optional): number of parallel processes to use when
            calculating feature matrix. Requires Dask if not equal to 1.

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

        progress_callback (callable): function to be called with incremental progress updates.
            Has the following parameters:

                update: percentage change (float between 0 and 100) in progress since last call
                progress_percent: percentage (float between 0 and 100) of total computation completed
                time_elapsed: total time in seconds that has elapsed since start of call

        include_cutoff_time (bool): Include data at cutoff times in feature calculations. Defaults to ``True``.

    Returns:
        pd.DataFrame: The feature matrix.
    """
    assert (
        isinstance(features, list)
        and features != []
        and all([isinstance(feature, FeatureBase) for feature in features])
    ), "features must be a non-empty list of features"

    # handle loading entityset
    from featuretools.entityset.entityset import EntitySet

    if not isinstance(entityset, EntitySet):
        if dataframes is not None:
            entityset = EntitySet("entityset", dataframes, relationships)
        else:
            raise TypeError("No dataframes or valid EntitySet provided")

    if entityset.dataframe_type == Library.DASK:
        if approximate:
            msg = "Using approximate is not supported with Dask dataframes"
            raise ValueError(msg)
        if training_window:
            msg = "Using training_window is not supported with Dask dataframes"
            raise ValueError(msg)

    target_dataframe = entityset[features[0].dataframe_name]

    cutoff_time = _validate_cutoff_time(cutoff_time, target_dataframe)
    entityset._check_time_indexes()

    if isinstance(cutoff_time, pd.DataFrame):
        if instance_ids:
            msg = "Passing 'instance_ids' is valid only if 'cutoff_time' is a single value or None - ignoring"
            warnings.warn(msg)
        pass_columns = [
            col for col in cutoff_time.columns if col not in ["instance_id", "time"]
        ]
        # make sure dtype of instance_id in cutoff time
        # is same as column it references
        target_dataframe = features[0].dataframe
        ltype = target_dataframe.ww.logical_types[target_dataframe.ww.index]
        cutoff_time.ww.init(logical_types={"instance_id": ltype})
    else:
        pass_columns = []
        if cutoff_time is None:
            if entityset.time_type == "numeric":
                cutoff_time = np.inf
            else:
                cutoff_time = datetime.now()

        if instance_ids is None:
            index_col = target_dataframe.ww.index
            df = entityset._handle_time(
                dataframe_name=target_dataframe.ww.name,
                df=target_dataframe,
                time_last=cutoff_time,
                training_window=training_window,
                include_cutoff_time=include_cutoff_time,
            )
            instance_ids = df[index_col]

        if is_instance(instance_ids, dd, "Series"):
            instance_ids = instance_ids.compute()
        elif is_instance(instance_ids, ps, "Series"):
            instance_ids = instance_ids.to_pandas()

        # convert list or range object into series
        if not isinstance(instance_ids, pd.Series):
            instance_ids = pd.Series(instance_ids)

        cutoff_time = (cutoff_time, instance_ids)

    _check_cutoff_time_type(cutoff_time, entityset.time_type)

    # Approximate provides no benefit with a single cutoff time, so ignore it
    if isinstance(cutoff_time, tuple) and approximate is not None:
        msg = (
            "Using approximate with a single cutoff_time value or no cutoff_time "
            "provides no computational efficiency benefit"
        )
        warnings.warn(msg)
        cutoff_time = pd.DataFrame(
            {
                "instance_id": cutoff_time[1],
                "time": [cutoff_time[0]] * len(cutoff_time[1]),
            },
        )
        target_dataframe = features[0].dataframe
        ltype = target_dataframe.ww.logical_types[target_dataframe.ww.index]
        cutoff_time.ww.init(logical_types={"instance_id": ltype})

    feature_set = FeatureSet(features)

    # Get features to approximate
    if approximate is not None:
        approximate_feature_trie = gather_approximate_features(feature_set)
        # Make a new FeatureSet that ignores approximated features
        feature_set = FeatureSet(
            features,
            approximate_feature_trie=approximate_feature_trie,
        )

    # Check if there are any non-approximated aggregation features
    no_unapproximated_aggs = True
    for feature in features:
        if isinstance(feature, AggregationFeature):
            # do not need to check if feature is in to_approximate since
            # only base features of direct features can be in to_approximate
            no_unapproximated_aggs = False
            break

        if approximate is not None:
            all_approx_features = {
                f for _, feats in feature_set.approximate_feature_trie for f in feats
            }
        else:
            all_approx_features = set()
        deps = feature.get_dependencies(deep=True, ignored=all_approx_features)
        for dependency in deps:
            if isinstance(dependency, AggregationFeature):
                no_unapproximated_aggs = False
                break

    cutoff_df_time_col = "time"
    target_time = "_original_time"

    if approximate is not None:
        # If there are approximated aggs, bin times
        binned_cutoff_time = bin_cutoff_times(cutoff_time, approximate)

        # Think about collisions: what if original time is a feature
        binned_cutoff_time.ww[target_time] = cutoff_time[cutoff_df_time_col]

        cutoff_time_to_pass = binned_cutoff_time

    else:
        cutoff_time_to_pass = cutoff_time

    if isinstance(cutoff_time, pd.DataFrame):
        cutoff_time_len = cutoff_time.shape[0]
    else:
        cutoff_time_len = len(cutoff_time[1])

    chunk_size = _handle_chunk_size(chunk_size, cutoff_time_len)
    tqdm_options = {
        "total": (cutoff_time_len / FEATURE_CALCULATION_PERCENTAGE),
        "bar_format": PBAR_FORMAT,
        "disable": True,
    }

    if verbose:
        tqdm_options.update({"disable": False})
    elif progress_callback is not None:
        # allows us to utilize progress_bar updates without printing to anywhere
        tqdm_options.update({"file": open(os.devnull, "w"), "disable": False})

    with make_tqdm_iterator(**tqdm_options) as progress_bar:
        if n_jobs != 1 or dask_kwargs is not None:
            feature_matrix = parallel_calculate_chunks(
                cutoff_time=cutoff_time_to_pass,
                chunk_size=chunk_size,
                feature_set=feature_set,
                approximate=approximate,
                training_window=training_window,
                save_progress=save_progress,
                entityset=entityset,
                n_jobs=n_jobs,
                no_unapproximated_aggs=no_unapproximated_aggs,
                cutoff_df_time_col=cutoff_df_time_col,
                target_time=target_time,
                pass_columns=pass_columns,
                progress_bar=progress_bar,
                dask_kwargs=dask_kwargs or {},
                progress_callback=progress_callback,
                include_cutoff_time=include_cutoff_time,
            )
        else:
            feature_matrix = calculate_chunk(
                cutoff_time=cutoff_time_to_pass,
                chunk_size=chunk_size,
                feature_set=feature_set,
                approximate=approximate,
                training_window=training_window,
                save_progress=save_progress,
                entityset=entityset,
                no_unapproximated_aggs=no_unapproximated_aggs,
                cutoff_df_time_col=cutoff_df_time_col,
                target_time=target_time,
                pass_columns=pass_columns,
                progress_bar=progress_bar,
                progress_callback=progress_callback,
                include_cutoff_time=include_cutoff_time,
            )

        # ensure rows are sorted by input order
        if isinstance(feature_matrix, pd.DataFrame):
            if isinstance(cutoff_time, pd.DataFrame):
                feature_matrix = feature_matrix.ww.reindex(
                    pd.MultiIndex.from_frame(
                        cutoff_time[["instance_id", "time"]],
                        names=feature_matrix.index.names,
                    ),
                )
            else:
                # Maintain index dtype
                index_dtype = feature_matrix.index.get_level_values(0).dtype
                feature_matrix = feature_matrix.ww.reindex(
                    cutoff_time[1].astype(index_dtype),
                    level=0,
                )
            if not cutoff_time_in_index:
                feature_matrix.ww.reset_index(level="time", drop=True, inplace=True)

        if save_progress and os.path.exists(os.path.join(save_progress, "temp")):
            shutil.rmtree(os.path.join(save_progress, "temp"))

        # force to 100% since we saved last 5 percent
        previous_progress = progress_bar.n
        progress_bar.update(progress_bar.total - progress_bar.n)

        if progress_callback is not None:
            (
                update,
                progress_percent,
                time_elapsed,
            ) = update_progress_callback_parameters(progress_bar, previous_progress)
            progress_callback(update, progress_percent, time_elapsed)

        progress_bar.refresh()

    return feature_matrix


def calculate_chunk(
    cutoff_time,
    chunk_size,
    feature_set,
    entityset,
    approximate,
    training_window,
    save_progress,
    no_unapproximated_aggs,
    cutoff_df_time_col,
    target_time,
    pass_columns,
    progress_bar=None,
    progress_callback=None,
    include_cutoff_time=True,
    schema=None,
):
    if not isinstance(feature_set, FeatureSet):
        feature_set = cloudpickle.loads(feature_set)  # pragma: no cover

    feature_matrix = []
    if no_unapproximated_aggs and approximate is not None:
        if entityset.time_type == "numeric":
            group_time = np.inf
        else:
            group_time = datetime.now()

    if isinstance(cutoff_time, tuple):
        update_progress_callback = None
        if progress_bar is not None:

            def update_progress_callback(done):
                previous_progress = progress_bar.n
                progress_bar.update(done * len(cutoff_time[1]))
                if progress_callback is not None:
                    (
                        update,
                        progress_percent,
                        time_elapsed,
                    ) = update_progress_callback_parameters(
                        progress_bar,
                        previous_progress,
                    )
                    progress_callback(update, progress_percent, time_elapsed)

        time_last = cutoff_time[0]
        ids = cutoff_time[1]
        calculator = FeatureSetCalculator(
            entityset,
            feature_set,
            time_last,
            training_window=training_window,
        )
        _feature_matrix = calculator.run(
            ids,
            progress_callback=update_progress_callback,
            include_cutoff_time=include_cutoff_time,
        )
        if isinstance(_feature_matrix, pd.DataFrame):
            time_index = pd.Index([time_last] * len(ids), name="time")
            _feature_matrix = _feature_matrix.set_index(time_index, append=True)
        feature_matrix.append(_feature_matrix)

    else:
        if schema:
            cutoff_time.ww.init_with_full_schema(schema=schema)  # pragma: no cover
        for _, group in cutoff_time.groupby(cutoff_df_time_col):
            # if approximating, calculate the approximate features
            if approximate is not None:
                group.ww.init(schema=cutoff_time.ww.schema, validate=False)
                precalculated_features_trie = approximate_features(
                    feature_set,
                    group,
                    window=approximate,
                    entityset=entityset,
                    training_window=training_window,
                    include_cutoff_time=include_cutoff_time,
                )
            else:
                precalculated_features_trie = None

            @save_csv_decorator(save_progress)
            def calc_results(
                time_last,
                ids,
                precalculated_features=None,
                training_window=None,
                include_cutoff_time=True,
            ):
                update_progress_callback = None

                if progress_bar is not None:

                    def update_progress_callback(done):
                        previous_progress = progress_bar.n
                        progress_bar.update(done * group.shape[0])
                        if progress_callback is not None:
                            (
                                update,
                                progress_percent,
                                time_elapsed,
                            ) = update_progress_callback_parameters(
                                progress_bar,
                                previous_progress,
                            )
                            progress_callback(update, progress_percent, time_elapsed)

                calculator = FeatureSetCalculator(
                    entityset,
                    feature_set,
                    time_last,
                    training_window=training_window,
                    precalculated_features=precalculated_features,
                )
                matrix = calculator.run(
                    ids,
                    progress_callback=update_progress_callback,
                    include_cutoff_time=include_cutoff_time,
                )

                return matrix

            # if all aggregations have been approximated, can calculate all together
            if no_unapproximated_aggs and approximate is not None:
                inner_grouped = [[group_time, group]]
            else:
                # if approximated features, set cutoff_time to unbinned time
                if precalculated_features_trie is not None:
                    group[cutoff_df_time_col] = group[target_time]

                inner_grouped = group.groupby(cutoff_df_time_col, sort=True)

            if chunk_size is not None:
                inner_grouped = _chunk_dataframe_groups(inner_grouped, chunk_size)

            for time_last, group in inner_grouped:
                # sort group by instance id
                ids = group["instance_id"].sort_values().values
                if no_unapproximated_aggs and approximate is not None:
                    window = None
                else:
                    window = training_window

                # calculate values for those instances at time time_last
                _feature_matrix = calc_results(
                    time_last,
                    ids,
                    precalculated_features=precalculated_features_trie,
                    training_window=window,
                    include_cutoff_time=include_cutoff_time,
                )

                if is_instance(_feature_matrix, (dd, ps), "DataFrame"):
                    id_name = _feature_matrix.columns[-1]
                else:
                    id_name = _feature_matrix.index.name

                # if approximate, merge feature matrix with group frame to get original
                # cutoff times and passed columns
                if approximate:
                    cols = [c for c in _feature_matrix.columns if c not in pass_columns]
                    indexer = group[["instance_id", target_time] + pass_columns]
                    _feature_matrix = _feature_matrix[cols].merge(
                        indexer,
                        right_on=["instance_id"],
                        left_index=True,
                        how="right",
                    )
                    _feature_matrix.set_index(
                        ["instance_id", target_time],
                        inplace=True,
                    )
                    _feature_matrix.index.set_names([id_name, "time"], inplace=True)
                    _feature_matrix.sort_index(level=1, kind="mergesort", inplace=True)
                else:
                    # all rows have same cutoff time. set time and add passed columns
                    num_rows = len(ids)
                    if len(pass_columns) > 0:
                        pass_through = group[
                            ["instance_id", cutoff_df_time_col] + pass_columns
                        ]
                        pass_through.rename(
                            columns={
                                "instance_id": id_name,
                                cutoff_df_time_col: "time",
                            },
                            inplace=True,
                        )
                    if isinstance(_feature_matrix, pd.DataFrame):
                        time_index = pd.Index([time_last] * num_rows, name="time")
                        _feature_matrix = _feature_matrix.set_index(
                            time_index,
                            append=True,
                        )
                        if len(pass_columns) > 0:
                            pass_through.set_index([id_name, "time"], inplace=True)
                            for col in pass_columns:
                                _feature_matrix[col] = pass_through[col]
                    elif is_instance(_feature_matrix, dd, "DataFrame") and (
                        len(pass_columns) > 0
                    ):
                        _feature_matrix["time"] = time_last
                        for col in pass_columns:
                            pass_df = dd.from_pandas(
                                pass_through[[id_name, "time", col]],
                                npartitions=_feature_matrix.npartitions,
                            )
                            _feature_matrix = _feature_matrix.merge(
                                pass_df,
                                how="outer",
                            )
                        _feature_matrix = _feature_matrix.drop(columns=["time"])
                    elif is_instance(_feature_matrix, ps, "DataFrame") and (
                        len(pass_columns) > 0
                    ):
                        _feature_matrix["time"] = time_last
                        for col in pass_columns:
                            pass_df = ps.from_pandas(
                                pass_through[[id_name, "time", col]],
                            )
                            _feature_matrix = _feature_matrix.merge(
                                pass_df,
                                how="outer",
                            )
                        _feature_matrix = _feature_matrix.drop(columns=["time"])
                feature_matrix.append(_feature_matrix)

    ww_init_kwargs = get_ww_types_from_features(
        feature_set.target_features,
        entityset,
        pass_columns,
        cutoff_time,
    )
    feature_matrix = init_ww_and_concat_fm(feature_matrix, ww_init_kwargs)
    return feature_matrix


def approximate_features(
    feature_set,
    cutoff_time,
    window,
    entityset,
    training_window=None,
    include_cutoff_time=True,
):
    """Given a set of features and cutoff_times to be passed to
    calculate_feature_matrix, calculates approximate values of some features
    to speed up calculations.  Cutoff times are sorted into
    window-sized buckets and the approximate feature values are only calculated
    at one cutoff time for each bucket.


    ..note:: this only approximates DirectFeatures of AggregationFeatures, on
        the target dataframe. In future versions, it may also be possible to
        approximate these features on other top-level dataframes

    Args:
        cutoff_time (pd.DataFrame): specifies what time to calculate
            the features for each instance at. The resulting feature matrix will use data
            up to and including the cutoff_time. A DataFrame with
            'instance_id' and 'time' columns.

        window (Timedelta or str): frequency to group instances with similar
            cutoff times by for features with costly calculations. For example,
            if bucket is 24 hours, all instances with cutoff times on the same
            day will use the same calculation for expensive features.

        entityset (:class:`.EntitySet`): An already initialized entityset.

        feature_set (:class:`.FeatureSet`): The features to be calculated.

        training_window (`Timedelta`, optional):
            Window defining how much older than the cutoff time data
            can be to be included when calculating the feature. If None, all older data is used.

        include_cutoff_time (bool):
            If True, data at cutoff times are included in feature calculations.

    """
    approx_fms_trie = Trie(path_constructor=RelationshipPath)

    target_time_colname = "target_time"
    cutoff_time.ww[target_time_colname] = cutoff_time["time"]
    approx_cutoffs = bin_cutoff_times(cutoff_time, window)
    cutoff_df_time_col = "time"
    cutoff_df_instance_col = "instance_id"
    # should this order be by dependencies so that calculate_feature_matrix
    # doesn't skip approximating something?
    for relationship_path, approx_feature_names in feature_set.approximate_feature_trie:
        if not approx_feature_names:
            continue

        (
            cutoffs_with_approx_e_ids,
            new_approx_dataframe_index_col,
        ) = _add_approx_dataframe_index_col(
            entityset,
            feature_set.target_df_name,
            approx_cutoffs.copy(),
            relationship_path,
        )

        # Select only columns we care about
        columns_we_want = [
            new_approx_dataframe_index_col,
            cutoff_df_time_col,
            target_time_colname,
        ]

        cutoffs_with_approx_e_ids = cutoffs_with_approx_e_ids[columns_we_want]
        cutoffs_with_approx_e_ids = cutoffs_with_approx_e_ids.drop_duplicates()
        cutoffs_with_approx_e_ids.dropna(
            subset=[new_approx_dataframe_index_col],
            inplace=True,
        )

        approx_features = [
            feature_set.features_by_name[name] for name in approx_feature_names
        ]
        if cutoffs_with_approx_e_ids.empty:
            approx_fm = gen_empty_approx_features_df(approx_features)
        else:
            cutoffs_with_approx_e_ids.sort_values(
                [cutoff_df_time_col, new_approx_dataframe_index_col],
                inplace=True,
            )
            # CFM assumes specific column names for cutoff_time argument
            rename = {new_approx_dataframe_index_col: cutoff_df_instance_col}
            cutoff_time_to_pass = cutoffs_with_approx_e_ids.rename(columns=rename)
            cutoff_time_to_pass = cutoff_time_to_pass[
                [cutoff_df_instance_col, cutoff_df_time_col]
            ]

            cutoff_time_to_pass.drop_duplicates(inplace=True)
            approx_fm = calculate_feature_matrix(
                approx_features,
                entityset,
                cutoff_time=cutoff_time_to_pass,
                training_window=training_window,
                approximate=None,
                cutoff_time_in_index=False,
                chunk_size=cutoff_time_to_pass.shape[0],
                include_cutoff_time=include_cutoff_time,
            )

        approx_fms_trie.get_node(relationship_path).value = approx_fm

    return approx_fms_trie


def scatter_warning(num_scattered_workers, num_workers):
    if num_scattered_workers != num_workers:
        scatter_warning = "EntitySet was only scattered to {} out of {} workers"
        logger.warning(scatter_warning.format(num_scattered_workers, num_workers))


def parallel_calculate_chunks(
    cutoff_time,
    chunk_size,
    feature_set,
    approximate,
    training_window,
    save_progress,
    entityset,
    n_jobs,
    no_unapproximated_aggs,
    cutoff_df_time_col,
    target_time,
    pass_columns,
    progress_bar,
    dask_kwargs=None,
    progress_callback=None,
    include_cutoff_time=True,
):
    import_or_raise(
        "distributed",
        "Dask must be installed to calculate feature matrix with n_jobs set to anything but 1",
    )
    from dask.base import tokenize
    from distributed import Future, as_completed

    client = None
    cluster = None
    try:
        client, cluster = create_client_and_cluster(
            n_jobs=n_jobs,
            dask_kwargs=dask_kwargs,
            entityset_size=entityset.__sizeof__(),
        )
        # scatter the entityset
        # denote future with leading underscore
        start = time.time()
        es_token = "EntitySet-{}".format(tokenize(entityset))
        if es_token in client.list_datasets():
            msg = "Using EntitySet persisted on the cluster as dataset {}"
            progress_bar.write(msg.format(es_token))
            _es = client.get_dataset(es_token)
        else:
            _es = client.scatter([entityset])[0]
            client.publish_dataset(**{_es.key: _es})

        # save features to a tempfile and scatter it
        pickled_feats = cloudpickle.dumps(feature_set)
        _saved_features = client.scatter(pickled_feats)
        client.replicate([_es, _saved_features])
        num_scattered_workers = len(
            client.who_has([Future(es_token)]).get(es_token, []),
        )
        num_workers = len(client.scheduler_info()["workers"].values())

        schema = None
        if isinstance(cutoff_time, pd.DataFrame):
            schema = cutoff_time.ww.schema
            chunks = cutoff_time.groupby(cutoff_df_time_col)
            cutoff_time_len = cutoff_time.shape[0]
        else:
            chunks = cutoff_time
            cutoff_time_len = len(cutoff_time[1])

        if not chunk_size:
            chunk_size = _handle_chunk_size(1.0 / num_workers, cutoff_time_len)

        chunks = _chunk_dataframe_groups(chunks, chunk_size)

        chunks = [df for _, df in chunks]

        if len(chunks) < num_workers:  # pragma: no cover
            chunk_warning = (
                "Fewer chunks ({}), than workers ({}) consider reducing the chunk size"
            )
            warning_string = chunk_warning.format(len(chunks), num_workers)
            progress_bar.write(warning_string)

        scatter_warning(num_scattered_workers, num_workers)
        end = time.time()
        scatter_time = round(end - start)

        # if enabled, reset timer after scatter for better time remaining estimates
        if not progress_bar.disable:
            progress_bar.reset()

        scatter_string = "EntitySet scattered to {} workers in {} seconds"
        progress_bar.write(scatter_string.format(num_scattered_workers, scatter_time))
        # map chunks
        # TODO: consider handling task submission dask kwargs
        _chunks = client.map(
            calculate_chunk,
            chunks,
            feature_set=_saved_features,
            chunk_size=None,
            entityset=_es,
            approximate=approximate,
            training_window=training_window,
            save_progress=save_progress,
            no_unapproximated_aggs=no_unapproximated_aggs,
            cutoff_df_time_col=cutoff_df_time_col,
            target_time=target_time,
            pass_columns=pass_columns,
            progress_bar=None,
            progress_callback=progress_callback,
            include_cutoff_time=include_cutoff_time,
            schema=schema,
        )

        feature_matrix = []
        iterator = as_completed(_chunks).batches()
        for batch in iterator:
            results = client.gather(batch)
            for result in results:
                feature_matrix.append(result)
                previous_progress = progress_bar.n
                progress_bar.update(result.shape[0])
                if progress_callback is not None:
                    (
                        update,
                        progress_percent,
                        time_elapsed,
                    ) = update_progress_callback_parameters(
                        progress_bar,
                        previous_progress,
                    )
                    progress_callback(update, progress_percent, time_elapsed)

    except Exception:
        raise
    finally:
        if client is not None:
            client.close()

        if "cluster" not in dask_kwargs and cluster is not None:
            cluster.close()  # pragma: no cover

    ww_init_kwargs = get_ww_types_from_features(
        feature_set.target_features,
        entityset,
        pass_columns,
        cutoff_time,
    )
    feature_matrix = init_ww_and_concat_fm(feature_matrix, ww_init_kwargs)
    return feature_matrix


def _add_approx_dataframe_index_col(es, target_dataframe_name, cutoffs, path):
    """
    Add a column to the cutoff df linking it to the dataframe at the end of the
    path.

    Return the updated cutoff df and the name of this column. The name will
    consist of the columns which were joined through.
    """
    last_child_col = "instance_id"
    last_parent_col = es[target_dataframe_name].ww.index

    for _, relationship in path:
        child_cols = [last_parent_col, relationship._child_column_name]
        child_df = es[relationship.child_name][child_cols]

        # Rename relationship.child_column to include the columns we have
        # joined through.
        new_col_name = "%s.%s" % (last_child_col, relationship._child_column_name)
        to_rename = {relationship._child_column_name: new_col_name}
        child_df = child_df.rename(columns=to_rename)
        cutoffs = cutoffs.merge(
            child_df,
            left_on=last_child_col,
            right_on=last_parent_col,
        )

        # These will be used in the next iteration.
        last_child_col = new_col_name
        last_parent_col = relationship._parent_column_name

    return cutoffs, new_col_name


def _chunk_dataframe_groups(grouped, chunk_size):
    """chunks a grouped dataframe into groups no larger than chunk_size"""
    if isinstance(grouped, tuple):
        for i in range(0, len(grouped[1]), chunk_size):
            yield None, (grouped[0], grouped[1].iloc[i : i + chunk_size])
    else:
        for group_key, group_df in grouped:
            for i in range(0, len(group_df), chunk_size):
                yield group_key, group_df.iloc[i : i + chunk_size]


def _handle_chunk_size(chunk_size, total_size):
    if chunk_size is not None:
        assert chunk_size > 0, "Chunk size must be greater than 0"

        if chunk_size < 1:
            chunk_size = math.ceil(chunk_size * total_size)

        chunk_size = int(chunk_size)

    return chunk_size


def update_progress_callback_parameters(progress_bar, previous_progress):
    update = (progress_bar.n - previous_progress) / progress_bar.total * 100
    progress_percent = (progress_bar.n / progress_bar.total) * 100
    time_elapsed = progress_bar.format_dict["elapsed"]
    return (update, progress_percent, time_elapsed)


def init_ww_and_concat_fm(feature_matrix, ww_init_kwargs):
    cols_to_check = {
        col
        for col, ltype in ww_init_kwargs["logical_types"].items()
        if isinstance(ltype, (Age, Boolean, Integer))
    }
    replacement_type = {
        "age": AgeNullable(),
        "boolean": BooleanNullable(),
        "integer": IntegerNullable(),
    }
    for fm in feature_matrix:
        updated_cols = set()
        for col in cols_to_check:
            # Only convert types for pandas if null values are present
            # Always convert for Dask/Spark to avoid pulling data into memory for null check
            is_pandas_df_with_null = (
                isinstance(fm, pd.DataFrame) and fm[col].isnull().any()
            )
            is_dask_df = is_instance(fm, dd, "DataFrame")
            is_spark_df = is_instance(fm, ps, "DataFrame")
            if is_pandas_df_with_null or is_dask_df or is_spark_df:
                current_type = ww_init_kwargs["logical_types"][col].type_string
                ww_init_kwargs["logical_types"][col] = replacement_type[current_type]
                updated_cols.add(col)
        cols_to_check = cols_to_check - updated_cols
        fm.ww.init(**ww_init_kwargs)

    if any(is_instance(fm, dd, "DataFrame") for fm in feature_matrix):
        feature_matrix = dd.concat(feature_matrix)
    elif any(is_instance(fm, ps, "DataFrame") for fm in feature_matrix):
        feature_matrix = ps.concat(feature_matrix)
    else:
        feature_matrix = pd.concat(feature_matrix)

    feature_matrix.ww.init(**ww_init_kwargs)
    return feature_matrix
