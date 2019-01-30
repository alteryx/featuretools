# -*- coding: utf-8 -*-
from __future__ import absolute_import

import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin

from featuretools.computational_backends import calculate_feature_matrix
from featuretools.synthesis import dfs


class DFSTransformer(TransformerMixin):
    """Transformer using Scikit-Learn interface for Pipeline uses.
    """

    def __init__(self,
                 entities=None,
                 relationships=None,
                 entityset=None,
                 target_entity=None,
                 agg_primitives=None,
                 trans_primitives=None,
                 allowed_paths=None,
                 max_depth=2,
                 ignore_entities=None,
                 ignore_variables=None,
                 seed_features=None,
                 drop_contains=None,
                 drop_exact=None,
                 where_primitives=None,
                 max_features=-1,
                 verbose=False,
                 profile=False):
        """Creates Transformer

        Args:

            entities (dict[str -> tuple(pd.DataFrame, str, str)]): Dictionary
                of entities. Entries take the format
                {entity id -> (dataframe, id column, (time_column))}.

            relationships (list[(str, str, str, str)]): List of relationships
                between entities. List items are a tuple with the format
                (parent entity id, parent variable, child entity id, child
                variable).

            entityset (EntitySet): An already initialized entityset. Required
                if entities and relationships are not defined.

            target_entity (str): Entity id of entity on which to make
                predictions.

            agg_primitives (list[str or AggregationPrimitive], optional): List
                of Aggregation Feature types to apply.

                    Default: ["sum", "std", "max", "skew", "min", "mean",
                              "count", "percent_true", "n_unique", "mode"]

            trans_primitives (list[str or TransformPrimitive], optional):
                List of Transform Feature functions to apply.

                    Default: ["day", "year", "month", "weekday", "haversine",
                              "num_words", "num_characters"]

            allowed_paths (list[list[str]]): Allowed entity paths on which to
                make features.

            max_depth (int) : Maximum allowed depth of features.

            ignore_entities (list[str], optional): List of entities to
                blacklist when creating features.

            ignore_variables (dict[str -> list[str]], optional): List of
                specific variables within each entity to blacklist when
                creating features.

            seed_features (list[:class:`.FeatureBase`]): List of manually
                defined features to use.

            drop_contains (list[str], optional): Drop features
                that contains these strings in name.

            drop_exact (list[str], optional): Drop features that
                exactly match these strings in name.

            where_primitives (list[str or PrimitiveBase], optional):
                List of Primitives names (or types) to apply with where
                clauses.

                    Default:

                        ["count"]

            max_features (int, optional) : Cap the number of generated features
                    to this number. If -1, no limit.

            profile (bool, optional): Enables profiling if True.

        Example:
            .. ipython:: python

                import featuretools as ft
                import pandas as pd

                from sklearn.pipeline import Pipeline
                from sklearn.ensemble import ExtraTreesClassifier

                # Get examle data
                n_customers = 3
                es = ft.demo.load_mock_customer(return_entityset=True, n_customers=5)
                y = [True, False, True]

                # Build dataset
                pipeline = Pipeline(steps=[
                    ('ft', ft.wrappers.DFSTransformer(entityset=es,
                                                      target_entity="customers",
                                                      max_features=3)),
                    ('et', ExtraTreesClassifier(n_estimators=100))
                ])

                # Fit and predict
                pipeline.fit([1, 2, 3], y=y) # fit on first 3 customers
                pipeline.predict_proba([4,5]) # predict probability of each class on last 2
                pipeline.predict([4,5]) # predict on last 2

                # Same as above, but using cutoff times
                ct = pd.DataFrame()
                ct['customer_id'] = [1, 2, 3, 4, 5]
                ct['time'] = pd.to_datetime(['2014-1-1 04:00',
                                             '2014-1-2 17:20',
                                             '2014-1-4 09:53',
                                             '2014-1-4 13:48',
                                             '2014-1-5 15:32'])

                pipeline.fit(ct.head(3), y=y)
                pipeline.predict_proba(ct.tail(2))
                pipeline.predict(ct.tail(2))

        """
        self.feature_defs = []
        self.entities = entities
        self.relationships = relationships
        self.entityset = entityset
        self.target_entity = target_entity
        self.agg_primitives = agg_primitives
        self.trans_primitives = trans_primitives
        self.allowed_paths = allowed_paths
        self.max_depth = max_depth
        self.ignore_entities = ignore_entities
        self.ignore_variables = ignore_variables
        self.seed_features = seed_features
        self.drop_contains = drop_contains
        self.drop_exact = drop_exact
        self.where_primitives = where_primitives
        self.max_features = max_features
        self.verbose = verbose
        self.profile = profile

    def fit(self, cuttof_time_ids, y=None):
        """Wrapper for DFS

            Calculates a feature matrix and features given a dictionary of
            entities and a list of relationships.

            Args:
                cuttof_time_ids (list | DataFrame): Instances filtered to
                    calculate features on.

            See Also:
                :func:`synthesis.dfs`
        """
        if isinstance(cuttof_time_ids, (list, np.ndarray, pd.Series)):
            self.feature_defs = dfs(entities=self.entities,
                                    relationships=self.relationships,
                                    entityset=self.entityset,
                                    target_entity=self.target_entity,
                                    instance_ids=cuttof_time_ids,
                                    agg_primitives=self.agg_primitives,
                                    trans_primitives=self.trans_primitives,
                                    allowed_paths=self.allowed_paths,
                                    max_depth=self.max_depth,
                                    ignore_entities=self.ignore_entities,
                                    ignore_variables=self.ignore_variables,
                                    seed_features=self.seed_features,
                                    drop_contains=self.drop_contains,
                                    drop_exact=self.drop_exact,
                                    where_primitives=self.where_primitives,
                                    max_features=self.max_features,
                                    features_only=True,
                                    verbose=self.verbose)

        elif isinstance(cuttof_time_ids, pd.DataFrame):
            self.feature_defs = dfs(entities=self.entities,
                                    relationships=self.relationships,
                                    entityset=self.entityset,
                                    target_entity=self.target_entity,
                                    cutoff_time=cuttof_time_ids,
                                    agg_primitives=self.agg_primitives,
                                    trans_primitives=self.trans_primitives,
                                    allowed_paths=self.allowed_paths,
                                    max_depth=self.max_depth,
                                    ignore_entities=self.ignore_entities,
                                    ignore_variables=self.ignore_variables,
                                    seed_features=self.seed_features,
                                    drop_contains=self.drop_contains,
                                    drop_exact=self.drop_exact,
                                    where_primitives=self.where_primitives,
                                    max_features=self.max_features,
                                    features_only=True,
                                    verbose=self.verbose)
        else:
            raise TypeError('instance_ids must be a list, np.ndarray, pd.Series, or pd.DataFrame')

        return self

    def transform(self, cuttof_time_ids):
        """Wrapper for calculate_feature_matix

            Calculates a matrix for a given set of instance ids and calculation
            times.

            Args:
                cuttof_time_ids (list | DataFrame): Instances filtered to
                    calculate features on.

            See Also:
                :func:`computational_backends.calculate_feature_matrix`
        """
        if isinstance(cuttof_time_ids, (list, np.ndarray, pd.Series)):
            X_transformed = calculate_feature_matrix(
                features=self.feature_defs,
                entityset=self.entityset,
                instance_ids=cuttof_time_ids,
                entities=self.entities,
                relationships=self.relationships,
                verbose=self.verbose,
                profile=self.profile)
            X_transformed = X_transformed.loc[cuttof_time_ids]
        elif isinstance(cuttof_time_ids, pd.DataFrame):
            ct = cuttof_time_ids
            X_transformed = calculate_feature_matrix(
                features=self.feature_defs,
                entityset=self.entityset,
                cutoff_time=cuttof_time_ids,
                entities=self.entities,
                relationships=self.relationships,
                verbose=self.verbose,
                profile=self.profile)
            X_transformed = X_transformed.loc[ct[ct.columns[0]]]
        else:
            raise TypeError('instance_ids must be a list or pd.DataFrame')
        return X_transformed

    def get_params(self, deep=True):
        out = {
            'entityset': self.entityset,
            'target_entity': self.target_entity,
            'entities': self.entities,
            'relationships': self.relationships,
            'agg_primitives': self.agg_primitives,
            'trans_primitives': self.trans_primitives,
            'allowed_paths': self.allowed_paths,
            'max_depth': self.max_depth,
            'ignore_entities': self.ignore_entities,
            'ignore_variables': self.ignore_variables,
            'seed_features': self.seed_features,
            'drop_contains': self.drop_contains,
            'drop_exact': self.drop_exact,
            'where_primitives': self.where_primitives,
            'max_features': self.max_features,
            'verbose': self.verbose,
            'profile': self.profile
        }
        return out
