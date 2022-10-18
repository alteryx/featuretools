import warnings

from featuretools.computational_backends import calculate_feature_matrix
from featuretools.entityset import EntitySet
from featuretools.exceptions import UnusedPrimitiveWarning
from featuretools.synthesis.deep_feature_synthesis import DeepFeatureSynthesis
from featuretools.synthesis.utils import _categorize_features, get_unused_primitives
from featuretools.utils import entry_point


@entry_point("featuretools_dfs")
def dfs(
    dataframes=None,
    relationships=None,
    entityset=None,
    target_dataframe_name=None,
    cutoff_time=None,
    instance_ids=None,
    agg_primitives=None,
    trans_primitives=None,
    groupby_trans_primitives=None,
    allowed_paths=None,
    max_depth=2,
    ignore_dataframes=None,
    ignore_columns=None,
    primitive_options=None,
    seed_features=None,
    drop_contains=None,
    drop_exact=None,
    where_primitives=None,
    max_features=-1,
    cutoff_time_in_index=False,
    save_progress=None,
    features_only=False,
    training_window=None,
    approximate=None,
    chunk_size=None,
    n_jobs=1,
    dask_kwargs=None,
    verbose=False,
    return_types=None,
    progress_callback=None,
    include_cutoff_time=True,
):
    """Calculates a feature matrix and features given a dictionary of dataframes
    and a list of relationships.


    Args:
        dataframes (dict[str -> tuple(DataFrame, str, str, dict[str -> str/Woodwork.LogicalType], dict[str->str/set], boolean)]):
            Dictionary of DataFrames. Entries take the format
            {dataframe name -> (dataframe, index column, time_index, logical_types, semantic_tags, make_index)}.
            Note that only the dataframe is required. If a Woodwork DataFrame is supplied, any other parameters
            will be ignored.

        relationships (list[(str, str, str, str)]): List of relationships
            between dataframes. List items are a tuple with the format
            (parent dataframe name, parent column, child dataframe name, child column).

        entityset (EntitySet): An already initialized entityset. Required if
            dataframes and relationships are not defined.

        target_dataframe_name (str): Name of dataframe on which to make predictions.

        cutoff_time (pd.DataFrame or Datetime or str): Specifies times at which to calculate
            the features for each instance. The resulting feature matrix will use data
            up to and including the cutoff_time. Can either be a DataFrame, a single
            value, or a string that can be parsed into a datetime. If a DataFrame is passed
            the instance ids for which to calculate features must be in a column with the
            same name as the target dataframe index or a column named `instance_id`.
            The cutoff time values in the DataFrame must be in a column with the same name as
            the target dataframe time index or a column named `time`. If the DataFrame has more
            than two columns, any additional columns will be added to the resulting feature
            matrix. If a single value is passed, this value will be used for all instances.

        instance_ids (list): List of instances on which to calculate features. Only
            used if cutoff_time is a single datetime.

        agg_primitives (list[str or AggregationPrimitive], optional): List of Aggregation
            Feature types to apply.

                Default: ["sum", "std", "max", "skew", "min", "mean", "count", "percent_true", "num_unique", "mode"]

        trans_primitives (list[str or TransformPrimitive], optional):
            List of Transform Feature functions to apply.

                Default: ["day", "year", "month", "weekday", "haversine", "num_words", "num_characters"]

        groupby_trans_primitives (list[str or TransformPrimitive], optional):
            list of Transform primitives to make GroupByTransformFeatures with

        allowed_paths (list[list[str]]): Allowed dataframe paths on which to make
            features.

        max_depth (int) : Maximum allowed depth of features.

        ignore_dataframes (list[str], optional): List of dataframes to
            blacklist when creating features.

        ignore_columns (dict[str -> list[str]], optional): List of specific
            columns within each dataframe to blacklist when creating features.

        primitive_options (list[dict[str or tuple[str] -> dict] or dict[str or tuple[str] -> dict, optional]):
            Specify options for a single primitive or a group of primitives.
            Lists of option dicts are used to specify options per input for primitives
            with multiple inputs. Each option ``dict`` can have the following keys:

            ``"include_dataframes"``
                List of dataframes to be included when creating features for
                the primitive(s). All other dataframes will be ignored
                (list[str]).
            ``"ignore_dataframes"``
                List of dataframes to be blacklisted when creating features
                for the primitive(s) (list[str]).
            ``"include_columns"``
                List of specific columns within each dataframe to include when
                creating features for the primitive(s). All other columns
                in a given dataframe will be ignored (dict[str -> list[str]]).
            ``"ignore_columns"``
                List of specific columns within each dataframe to blacklist
                when creating features for the primitive(s) (dict[str ->
                list[str]]).
            ``"include_groupby_dataframes"``
                List of dataframes to be included when finding groupbys. All
                other dataframes will be ignored (list[str]).
            ``"ignore_groupby_dataframes"``
                List of dataframes to blacklist when finding groupbys
                (list[str]).
            ``"include_groupby_columns"``
                List of specific columns within each dataframe to include as
                groupbys, if applicable. All other columns in each
                dataframe will be ignored (dict[str -> list[str]]).
            ``"ignore_groupby_columns"``
                List of specific columns within each dataframe to blacklist
                as groupbys (dict[str -> list[str]]).

        seed_features (list[:class:`.FeatureBase`]): List of manually defined
            features to use.

        drop_contains (list[str], optional): Drop features
            that contains these strings in name.

        drop_exact (list[str], optional): Drop features that
            exactly match these strings in name.

        where_primitives (list[str or PrimitiveBase], optional):
            List of Primitives names (or types) to apply with where clauses.

                Default:

                    ["count"]

        max_features (int, optional) : Cap the number of generated features to
                this number. If -1, no limit.

        features_only (bool, optional): If True, returns the list of
            features without calculating the feature matrix.

        cutoff_time_in_index (bool): If True, return a DataFrame with a MultiIndex
            where the second index is the cutoff time (first is instance id).
            DataFrame will be sorted by (time, instance_id).

        training_window (Timedelta or str, optional):
            Window defining how much time before the cutoff time data
            can be used when calculating features. If ``None`` , all data
            before cutoff time is used. Defaults to ``None``. Month and year
            units are not relative when Pandas Timedeltas are used. Relative
            units should be passed as a Featuretools Timedelta or a string.

        approximate (Timedelta): Bucket size to group instances with similar
            cutoff times by for features with costly calculations. For example,
            if bucket is 24 hours, all instances with cutoff times on the same
            day will use the same calculation for expensive features.

        save_progress (str, optional): Path to save intermediate computational results.

        n_jobs (int, optional): number of parallel processes to use when
            calculating feature matrix

        chunk_size (int or float or None or "cutoff time", optional): Number
            of rows of output feature matrix to calculate at time. If passed an
            integer greater than 0, will try to use that many rows per chunk.
            If passed a float value between 0 and 1 sets the chunk size to that
            percentage of all instances. If passed the string "cutoff time",
            rows are split per cutoff time.

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

        return_types (list[woodwork.ColumnSchema] or str, optional):
            List of ColumnSchemas defining the types of
            columns to return. If None, defaults to returning all
            numeric, categorical and boolean types. If given as
            the string 'all', returns all available types.

        progress_callback (callable): function to be called with incremental progress updates.
            Has the following parameters:

                update: percentage change (float between 0 and 100) in progress since last call
                progress_percent: percentage (float between 0 and 100) of total computation completed
                time_elapsed: total time in seconds that has elapsed since start of call

        include_cutoff_time (bool): Include data at cutoff times in feature calculations. Defaults to ``True``.

    Returns:
        list[:class:`.FeatureBase`], pd.DataFrame:
            The list of generated feature defintions, and the feature matrix.
            If ``features_only`` is ``True``, the feature matrix will not be generated.

    Examples:
        .. code-block:: python

            from featuretools.primitives import Mean
            # cutoff times per instance
            dataframes = {
                "sessions" : (session_df, "id"),
                "transactions" : (transactions_df, "id", "transaction_time")
            }
            relationships = [("sessions", "id", "transactions", "session_id")]
            feature_matrix, features = dfs(dataframes=dataframes,
                                           relationships=relationships,
                                           target_dataframe_name="transactions",
                                           cutoff_time=cutoff_times)
            feature_matrix

            features = dfs(dataframes=dataframes,
                           relationships=relationships,
                           target_dataframe_name="transactions",
                           features_only=True)
    """
    if not isinstance(entityset, EntitySet):
        entityset = EntitySet("dfs", dataframes, relationships)

    dfs_object = DeepFeatureSynthesis(
        target_dataframe_name,
        entityset,
        agg_primitives=agg_primitives,
        trans_primitives=trans_primitives,
        groupby_trans_primitives=groupby_trans_primitives,
        max_depth=max_depth,
        where_primitives=where_primitives,
        allowed_paths=allowed_paths,
        drop_exact=drop_exact,
        drop_contains=drop_contains,
        ignore_dataframes=ignore_dataframes,
        ignore_columns=ignore_columns,
        primitive_options=primitive_options,
        max_features=max_features,
        seed_features=seed_features,
    )

    features = dfs_object.build_features(verbose=verbose, return_types=return_types)

    trans, agg, groupby, where = _categorize_features(features)

    trans_unused = get_unused_primitives(trans_primitives, trans)
    agg_unused = get_unused_primitives(agg_primitives, agg)
    groupby_unused = get_unused_primitives(groupby_trans_primitives, groupby)
    where_unused = get_unused_primitives(where_primitives, where)

    unused_primitives = [trans_unused, agg_unused, groupby_unused, where_unused]
    if any(unused_primitives):
        warn_unused_primitives(unused_primitives)

    if features_only:
        return features

    assert (
        features != []
    ), "No features can be generated from the specified primitives. Please make sure the primitives you are using are compatible with the variable types in your data."

    feature_matrix = calculate_feature_matrix(
        features,
        entityset=entityset,
        cutoff_time=cutoff_time,
        instance_ids=instance_ids,
        training_window=training_window,
        approximate=approximate,
        cutoff_time_in_index=cutoff_time_in_index,
        save_progress=save_progress,
        chunk_size=chunk_size,
        n_jobs=n_jobs,
        dask_kwargs=dask_kwargs,
        verbose=verbose,
        progress_callback=progress_callback,
        include_cutoff_time=include_cutoff_time,
    )
    return feature_matrix, features


def warn_unused_primitives(unused_primitives):
    messages = [
        "  trans_primitives: {}\n",
        "  agg_primitives: {}\n",
        "  groupby_trans_primitives: {}\n",
        "  where_primitives: {}\n",
    ]
    unused_string = ""
    for primitives, message in zip(unused_primitives, messages):
        if primitives:
            unused_string += message.format(primitives)

    warning_msg = (
        "Some specified primitives were not used during DFS:\n{}".format(unused_string)
        + "This may be caused by a using a value of max_depth that is too small, not setting interesting values, "
        + "or it may indicate no compatible columns for the primitive were found in the data. If the DFS call "
        + "contained multiple instances of a primitive in the list above, none of them were used."
    )

    warnings.warn(warning_msg, UnusedPrimitiveWarning)
