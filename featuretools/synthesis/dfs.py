import pandas as pd

from .deep_feature_synthesis import DeepFeatureSynthesis

from featuretools.computational_backends import calculate_feature_matrix
from featuretools.entityset import EntitySet


def dfs(entities=None,
        relationships=None,
        entityset=None,
        target_entity=None,
        cutoff_time=None,
        instance_ids=None,
        agg_primitives=None,
        trans_primitives=None,
        allowed_paths=None,
        max_depth=None,
        ignore_entities=None,
        ignore_variables=None,
        seed_features=None,
        drop_contains=None,
        drop_exact=None,
        where_primitives=None,
        max_features=None,
        cutoff_time_in_index=False,
        save_progress=None,
        features_only=False,
        training_window=None,
        approximate=None,
        verbose=False):
    '''Calculates a feature matrix and features given a dictionary of entities
    and a list of relationships.


    Args:
        entities (dict[str: tuple(pd.DataFrame, str, str)]): dictionary of
            entities. Entries take the format
            {entity id: (dataframe, id column, (time_column))}

        relationships (list[(str, str, str, str)]): list of relationships
            between entities. List items are a tuple with the format
            (parent entity id, parent variable, child entity id, child variable)

        entityset (:class:`.EntitySet`): An already initialized entityset. Required if
            entities and relationships are not defined

        target_entity (str): id of entity to predict on

        cutoff_time (pd.DataFrame or Datetime): specifies what time to calculate
            the features for each instance at.  Can either be a DataFrame with
            'instance_id' and 'time' columns, DataFrame with the name of the
            index variable in the target entity and a time column, a list of values, or a single
            value to calculate for all instances.

        instance_ids (list): list of instances to calculate features on. Only
            used if cutoff_time is a single datetime.

        agg_primitives (list[:class:`features.AggregationPrimitive`], optional):
            list of Aggregation Feature types to apply.

            Default: [:class:`features.Sum`, :class:`features.Std`,
             :class:`features.Max`, :class:`features.Skew`,
             :class:`features.Min`, :class:`features.Mean`,
             :class:`features.Count`, :class:`features.PercentTrue`,
             :class:`features.NUnique`, :class:`features.Mode`]

        trans_primitives (list[:class:`features.TransformPrimitive`], optional):
            list of Transform Feature functions to apply.

            Default: [:class:`features.Day`, :class:`features.Year`,
             :class:`features.Month`, :class:`features.Weekday`]

        allowed_paths (list[list[str]]): Allowed entity paths to make
            features for

        max_depth (int) : maximum allowed depth of features

        ignore_entities (list[str], optional): List of entities to
            blacklist when creating features

        ignore_variables (dict[str : str], optional): List of specific
            variables within each entity to blacklist when creating features

        seed_features (list[:class:`.PrimitiveBase`]): List of manually defined
            features to use.

        drop_contains (list[str], optional): drop features
            that contains these strings in name

        drop_exact (list[str], optional): drop features that
            exactly match these strings in name


        where_primitives (list[:class:`features.AggregationPrimitive`], optional):
            list of Aggregation Feature types to apply with where clauses.

        max_features (int, optional) : Cap the number of generated features to
                this number. If -1, no limit.

        features_only (boolean, optional): if True, returns the list of
            features without calculating the feature matrix.

        cutoff_time_in_index (bool): If True, return a DataFrame with a MultiIndex
            where the second index is the cutoff time (first is instance id).
            DataFrame will be sorted by (time, instance_id).

        training_window (dict[str-> :class:`Timedelta`] or :class:`Timedelta`, optional):
            Window or windows defining how much older than the cutoff time data
            can be to be included when calculating the feature.  To specify
            which entities to apply windows to, use a dictionary mapping entity
            id -> Timedelta. If None, all older data is used.

        approximate (Timedelta): bucket size to group instances with similar
            cutoff times by for features with costly calculations. For example,
            if bucket is 24 hours, all instances with cutoff times on the same
            day will use the same calculation for expensive features.

        save_progress (Optional(str)): path to save intermediate computational results


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
                                      max_depth=max_depth,
                                      where_primitives=where_primitives,
                                      allowed_paths=allowed_paths,
                                      drop_exact=drop_exact,
                                      drop_contains=drop_contains,
                                      ignore_entities=ignore_entities,
                                      ignore_variables=ignore_variables,
                                      max_features=max_features,
                                      seed_features=seed_features)

    features = dfs_object.build_features(verbose=verbose)

    if features_only:
        return features

    if isinstance(cutoff_time, pd.DataFrame):
        feature_matrix = calculate_feature_matrix(features,
                                                  cutoff_time=cutoff_time,
                                                  training_window=training_window,
                                                  approximate=approximate,
                                                  cutoff_time_in_index=cutoff_time_in_index,
                                                  save_progress=save_progress,
                                                  verbose=verbose)
    else:
        feature_matrix = calculate_feature_matrix(features,
                                                  cutoff_time=cutoff_time,
                                                  instance_ids=instance_ids,
                                                  training_window=training_window,
                                                  approximate=approximate,
                                                  cutoff_time_in_index=cutoff_time_in_index,
                                                  save_progress=save_progress,
                                                  verbose=verbose)
    return feature_matrix, features
