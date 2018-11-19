from sklearn.base import TransformerMixin

from featuretools.synthesis import dfs
from featuretools.computational_backends import calculate_feature_matrix


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
                 max_depth=None,
                 ignore_entities=None,
                 ignore_variables=None,
                 seed_features=None,
                 drop_contains=None,
                 drop_exact=None,
                 where_primitives=None,
                 max_features=None,
                 save_progress=None,
                 training_window=None,
                 approximate=None,
                 chunk_size=None,
                 n_jobs=1,
                 dask_kwargs=None,
                 verbose=False,
                 profile=False):
        """Creates Transformer

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

            agg_primitives (list[str or AggregationPrimitive], optional): List of Aggregation
                Feature types to apply.

                    Default: ["sum", "std", "max", "skew", "min", "mean", "count", "percent_true", "n_unique", "mode"]

            trans_primitives (list[str or TransformPrimitive], optional):
                List of Transform Feature functions to apply.

                    Default: ["day", "year", "month", "weekday", "haversine", "num_words", "num_characters"]

            allowed_paths (list[list[str]]): Allowed entity paths on which to make
                features.

            max_depth (int) : Maximum allowed depth of features.

            ignore_entities (list[str], optional): List of entities to
                blacklist when creating features.

            ignore_variables (dict[str -> list[str]], optional): List of specific
                variables within each entity to blacklist when creating features.

            seed_features (list[:class:`.PrimitiveBase`]): List of manually defined
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

            chunk_size (int or float or None or "cutoff time", optionsal): Number
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

            profile (bool, optional): Enables profiling if True.

        Example:
            .. code-block:: python
                import featuretools as ft
                from sklearn.preprocessing import StandardScaler

                # Get examle data
                es = ft.demo.load_mock_customer(return_entityset=True)
                df = es['customers'].df

                # Build dataset
                pipeline = Pipeline(steps=[
                    ('ft', dfsTransfomer(entityset=es, target_entity="customers", max_features=2)),
                    ('sc', StandardScaler()),
                ])

                # Fit and transform
                pipeline.fit(df['customer_id'].values).transform(df['customer_id'].values)

        """
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
        self.save_progress = save_progress
        self.training_window = training_window
        self.approximate = approximate
        self.chunk_size = chunk_size
        self.n_jobs = n_jobs
        self.dask_kwargs = dask_kwargs
        self.verbose = verbose
        self.profile = profile

    def fit(self, entity_ids, y=None):
        """Wrapper for dfs

            Calculates a feature matrix and features given a dictionary of entities and a list of relationships.

            Args:
                entity_ids (list): List of instances to calculate features on.

            See Also:
                :func:`synthesis.dfs`
        """
        self.feature_defs =    dfs(entities=self.entities,
                                   relationships=self.relationships,
                                   entityset=self.entityset,
                                   target_entity=self.target_entity,
                                   instance_ids=entity_ids,
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
                                   save_progress=self.save_progress,
                                   features_only=True,
                                   training_window=self.training_window,
                                   approximate=self.approximate,
                                   chunk_size=self.chunk_size,
                                   n_jobs=self.n_jobs,
                                   dask_kwargs=self.dask_kwargs,
                                   verbose=self.verbose)
        return self

    def transform(self, entity_ids, y=None):
        """Wrapper for calculate_feature_matix

            Calculates a matrix for a given set of instance ids and calculation times.

            Args:
                entity_ids (list): List of instances to calculate features on.

            See Also:
                :func:`computational_backends.calculate_feature_matrix`
        """
        X_transformed =    calculate_feature_matrix(features=self.feature_defs,
                                                    entityset=self.entityset,
                                                    instance_ids=entity_ids,
                                                    entities=self.entities,
                                                    relationships=self.relationships,
                                                    training_window=self.training_window,
                                                    approximate=self.approximate,
                                                    save_progress=self.save_progress,
                                                    verbose=self.verbose,
                                                    chunk_size=self.chunk_size,
                                                    n_jobs=self.n_jobs,
                                                    dask_kwargs=self.dask_kwargs,
                                                    profile=self.profile)
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
            'save_progress': self.save_progress,
            'training_window': self.training_window,
            'approximate': self.approximate,
            'chunk_size': self.chunk_size,
            'n_jobs': self.n_jobs,
            'dask_kwargs': self.dask_kwargs,
            'verbose': self.verbose,
            'profile': self.profile
        }
        return out
