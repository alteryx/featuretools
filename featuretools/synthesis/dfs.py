import pandas as pd

from .deep_feature_synthesis import DeepFeatureSynthesis

from featuretools.computational_backends import calculate_feature_matrix
from featuretools.entityset import EntitySet
from featuretools.utils import entry_point


@entry_point('featuretools_dfs')
def dfs(entities=None,
        relationships=None,
        entityset=None,
        target_entity=None,
        cutoff_time=None,
        instance_ids=None,
        agg_primitives=None,
        trans_primitives=None,
        groupby_trans_primitives=None,
        allowed_paths=None,
        max_depth=2,
        ignore_entities=None,
        ignore_variables=None,
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
        return_variable_types=None):
    '''Calculates a feature matrix and features given a dictionary of entities
    and a list of relationships.


    Args:
        entities (dict[str -> tuple(pd.DataFrame, str, str)]): Dictionary of
            entities. Entries take the format
            {entity id -> (dataframe, id column, (time_column))}.

        relationships (list[(str, str, str, str)]): List of relationships
            between entities. List items are a tuple with the format
            (parent entity id, parent variable, child entity id, child variable).

        entityset (EntitySet): An already initialized entityset. Required if
            entities and relationships are not defined.

        target_entity (str): Entity id of entity on which to make predictions.

        cutoff_time (pd.DataFrame or Datetime): Specifies times at which to
            calculate each instance. Can either be a DataFrame with
            'instance_id' and 'time' columns, a DataFrame with the name of the
            index variable in the target entity and a time column, a
            list of values, or a single
            value to calculate for all instances. If the dataframe has more than
            two columns, any additional columns will be added to the resulting
            feature matrix.

        instance_ids (list): List of instances on which to calculate features. Only
            used if cutoff_time is a single datetime.

        agg_primitives (list[str or AggregationPrimitive], optional): List of Aggregation
            Feature types to apply.

                Default: ["sum", "std", "max", "skew", "min", "mean", "count", "percent_true", "n_unique", "mode"]

        trans_primitives (list[str or TransformPrimitive], optional):
            List of Transform Feature functions to apply.

                Default: ["day", "year", "month", "weekday", "haversine", "num_words", "num_characters"]

        groupby_trans_primitives (list[str or :class:`.primitives.TransformPrimitive`], optional):
            list of Transform primitives to make GroupByTransformFeatures with

        allowed_paths (list[list[str]]): Allowed entity paths on which to make
            features.

        max_depth (int) : Maximum allowed depth of features.

        ignore_entities (list[str], optional): List of entities to
            blacklist when creating features.

        ignore_variables (dict[str -> list[str]], optional): List of specific
            variables within each entity to blacklist when creating features.

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

        training_window (Timedelta, optional):
            Window defining how much older than the cutoff time data
            can be to be included when calculating the feature. If None, all older data is used.

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

        return_variable_types (list[Variable] or str, optional): Types of
                variables to return. If None, default to
                Numeric, Discrete, and Boolean. If given as
                the string 'all', use all available variable types.

    Examples:
        .. code-block:: python

            from featuretools.primitives import Mean
            # cutoff times per instance
            entities = {
                "sessions" : (session_df, "id"),
                "transactions" : (transactions_df, "id", "transaction_time")
            }
            relationships = [("sessions", "id", "transactions", "session_id")]
            feature_matrix, features = dfs(entities=entities,
                                           relationships=relationships,
                                           target_entity="transactions",
                                           cutoff_time=cutoff_times)
            feature_matrix

            features = dfs(entities=entities,
                           relationships=relationships,
                           target_entity="transactions",
                           features_only=True)
    '''
    if not isinstance(entityset, EntitySet):
        entityset = EntitySet("dfs", entities, relationships)

    dfs_object = DeepFeatureSynthesis(target_entity, entityset,
                                      agg_primitives=agg_primitives,
                                      trans_primitives=trans_primitives,
                                      groupby_trans_primitives=groupby_trans_primitives,
                                      max_depth=max_depth,
                                      where_primitives=where_primitives,
                                      allowed_paths=allowed_paths,
                                      drop_exact=drop_exact,
                                      drop_contains=drop_contains,
                                      ignore_entities=ignore_entities,
                                      ignore_variables=ignore_variables,
                                      max_features=max_features,
                                      seed_features=seed_features)

    features = dfs_object.build_features(
        verbose=verbose, return_variable_types=return_variable_types)

    if features_only:
        return features

    if isinstance(cutoff_time, pd.DataFrame):
        feature_matrix = calculate_feature_matrix(features,
                                                  entityset=entityset,
                                                  cutoff_time=cutoff_time,
                                                  training_window=training_window,
                                                  approximate=approximate,
                                                  cutoff_time_in_index=cutoff_time_in_index,
                                                  save_progress=save_progress,
                                                  chunk_size=chunk_size,
                                                  n_jobs=n_jobs,
                                                  dask_kwargs=dask_kwargs,
                                                  verbose=verbose)
    else:
        feature_matrix = calculate_feature_matrix(features,
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
                                                  verbose=verbose)
    return feature_matrix, features
