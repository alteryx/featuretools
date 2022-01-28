import logging
import warnings
from collections import defaultdict

from woodwork.column_schema import ColumnSchema
from woodwork.logical_types import Boolean, BooleanNullable

from featuretools import primitives
from featuretools.entityset.entityset import LTI_COLUMN_NAME
from featuretools.entityset.relationship import RelationshipPath
from featuretools.feature_base import (
    AggregationFeature,
    DirectFeature,
    GroupByTransformFeature,
    IdentityFeature,
    TransformFeature
)
from featuretools.feature_base.utils import is_valid_input
from featuretools.primitives.base import (
    AggregationPrimitive,
    PrimitiveBase,
    TransformPrimitive
)
from featuretools.primitives.options_utils import (
    filter_groupby_matches_by_options,
    filter_matches_by_options,
    generate_all_primitive_options,
    ignore_dataframe_for_primitive
)
from featuretools.utils.gen_utils import Library, camel_and_title_to_snake

logger = logging.getLogger('featuretools')


class DeepFeatureSynthesis(object):
    """Automatically produce features for a target dataframe in an Entityset.

        Args:
            target_dataframe_name (str): Name of dataframe for which to build features.

            entityset (EntitySet): Entityset for which to build features.

            agg_primitives (list[str or :class:`.primitives.`], optional):
                list of Aggregation Feature types to apply.

                Default: ["sum", "std", "max", "skew", "min", "mean", "count", "percent_true", "num_unique", "mode"]

            trans_primitives (list[str or :class:`.primitives.TransformPrimitive`], optional):
                list of Transform primitives to use.

                Default: ["day", "year", "month", "weekday", "haversine", "num_words", "num_characters"]

            where_primitives (list[str or :class:`.primitives.PrimitiveBase`], optional):
                only add where clauses to these types of Primitives

                Default:

                    ["count"]

            groupby_trans_primitives (list[str or :class:`.primitives.TransformPrimitive`], optional):
                list of Transform primitives to make GroupByTransformFeatures with

            max_depth (int, optional) : maximum allowed depth of features.
                Default: 2. If -1, no limit.

            max_features (int, optional) : Cap the number of generated features to
                this number. If -1, no limit.

            allowed_paths (list[list[str]], optional): Allowed dataframe paths to make
                features for. If None, use all paths.

            ignore_dataframes (list[str], optional): List of dataframes to
                blacklist when creating features. If None, use all dataframes.

            ignore_columns (dict[str -> list[str]], optional): List of specific
                columns within each dataframe to blacklist when creating features.
                If None, use all columns.

            seed_features (list[:class:`.FeatureBase`], optional): List of manually
                defined features to use.

            drop_contains (list[str], optional): Drop features
                that contains these strings in name.

            drop_exact (list[str], optional): Drop features that
                exactly match these strings in name.

            where_stacking_limit (int, optional): Cap the depth of the where features.
                Default: 1

            primitive_options (dict[str or tuple[str] or PrimitiveBase -> dict or list[dict]], optional):
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
        """

    def __init__(self,
                 target_dataframe_name,
                 entityset,
                 agg_primitives=None,
                 trans_primitives=None,
                 where_primitives=None,
                 groupby_trans_primitives=None,
                 max_depth=2,
                 max_features=-1,
                 allowed_paths=None,
                 ignore_dataframes=None,
                 ignore_columns=None,
                 primitive_options=None,
                 seed_features=None,
                 drop_contains=None,
                 drop_exact=None,
                 where_stacking_limit=1):

        if target_dataframe_name not in entityset.dataframe_dict:
            es_name = entityset.id or 'entity set'
            msg = 'Provided target dataframe %s does not exist in %s' % (target_dataframe_name, es_name)
            raise KeyError(msg)

        # need to change max_depth to None because DFs terminates when  <0
        if max_depth == -1:
            max_depth = None

        # if just one dataframe, set max depth to 1 (transform stacking rule)
        if len(entityset.dataframe_dict) == 1 and (max_depth is None or max_depth > 1):
            warnings.warn("Only one dataframe in entityset, changing max_depth to "
                          "1 since deeper features cannot be created")
            max_depth = 1

        self.max_depth = max_depth

        self.max_features = max_features

        self.allowed_paths = allowed_paths
        if self.allowed_paths:
            self.allowed_paths = set()
            for path in allowed_paths:
                self.allowed_paths.add(tuple(path))

        if ignore_dataframes is None:
            self.ignore_dataframes = set()
        else:
            if not isinstance(ignore_dataframes, list):
                raise TypeError('ignore_dataframes must be a list')
            assert target_dataframe_name not in ignore_dataframes,\
                "Can't ignore target_dataframe!"
            self.ignore_dataframes = set(ignore_dataframes)

        self.ignore_columns = defaultdict(set)
        if ignore_columns is not None:
            # check if ignore_columns is not {str: list}
            if not all(isinstance(i, str) for i in ignore_columns.keys()) or not all(isinstance(i, list) for i in ignore_columns.values()):
                raise TypeError('ignore_columns should be dict[str -> list]')
            # check if list values are all of type str
            elif not all(all(isinstance(v, str) for v in value) for value in ignore_columns.values()):
                raise TypeError('list values should be of type str')
            for df_name, cols in ignore_columns.items():
                self.ignore_columns[df_name] = set(cols)
        self.target_dataframe_name = target_dataframe_name
        self.es = entityset

        for library in Library:
            if library.value == self.es.dataframe_type:
                df_library = library
                break

        if agg_primitives is None:
            agg_primitives = [p for p in primitives.get_default_aggregation_primitives() if df_library in p.compatibility]
        self.agg_primitives = []
        self.agg_primitives = sorted([check_primitive(p, "aggregation") for p in agg_primitives])

        if trans_primitives is None:
            trans_primitives = [p for p in primitives.get_default_transform_primitives() if df_library in p.compatibility]
        self.trans_primitives = sorted([check_primitive(p, "transform") for p in trans_primitives])

        if where_primitives is None:
            where_primitives = [primitives.Count]
        self.where_primitives = sorted([check_primitive(p, "where") for p in where_primitives])

        if groupby_trans_primitives is None:
            groupby_trans_primitives = []
        self.groupby_trans_primitives = sorted([check_primitive(p, "groupby transform") for p in groupby_trans_primitives])

        if primitive_options is None:
            primitive_options = {}
        all_primitives = self.trans_primitives + self.agg_primitives + \
            self.where_primitives + self.groupby_trans_primitives
        bad_primitives = [prim.name for prim in all_primitives if df_library not in prim.compatibility]
        if bad_primitives:
            msg = 'Selected primitives are incompatible with {} EntitySets: {}'
            raise ValueError(msg.format(df_library.value, ', '.join(bad_primitives)))

        self.primitive_options, self.ignore_dataframes, self.ignore_columns =\
            generate_all_primitive_options(all_primitives,
                                           primitive_options,
                                           self.ignore_dataframes,
                                           self.ignore_columns,
                                           self.es)
        self.seed_features = sorted(seed_features or [], key=lambda f: f.unique_name())
        self.drop_exact = drop_exact or []
        self.drop_contains = drop_contains or []
        self.where_stacking_limit = where_stacking_limit

    def build_features(self, return_types=None, verbose=False):
        """Automatically builds feature definitions for target
            dataframe using Deep Feature Synthesis algorithm

        Args:
            return_types (list[woodwork.ColumnSchema] or str, optional):
                List of ColumnSchemas defining the types of
                columns to return. If None, defaults to returning all
                numeric, categorical and boolean types. If given as
                the string 'all', use all available return types.

            verbose (bool, optional): If True, print progress.

        Returns:
            list[BaseFeature]: Returns a list of
                features for target dataframe, sorted by feature depth
                (shallow first).
        """
        all_features = {}

        self.where_clauses = defaultdict(set)

        if return_types is None:
            return_types = [ColumnSchema(semantic_tags=['numeric']),
                            ColumnSchema(semantic_tags=['category']),
                            ColumnSchema(logical_type=Boolean),
                            ColumnSchema(logical_type=BooleanNullable)]
        elif return_types == 'all':
            pass
        else:
            msg = "return_types must be a list, or 'all'"
            assert isinstance(return_types, list), msg

        self._run_dfs(self.es[self.target_dataframe_name], RelationshipPath([]),
                      all_features, max_depth=self.max_depth)

        new_features = list(all_features[self.target_dataframe_name].values())

        def filt(f):
            # remove identity features of the ID field of the target dataframe
            if (isinstance(f, IdentityFeature) and
                    f.dataframe_name == self.target_dataframe_name and
                    f.column_name == self.es[self.target_dataframe_name].ww.index):
                return False

            return True

        # filter out features with undesired return types
        if return_types != 'all':
            new_features = [
                f for f in new_features
                if any(is_valid_input(f.column_schema, schema) for schema in return_types)]
        new_features = list(filter(filt, new_features))

        new_features.sort(key=lambda f: f.get_depth())

        new_features = self._filter_features(new_features)

        if self.max_features > 0:
            new_features = new_features[:self.max_features]

        if verbose:
            print("Built {} features".format(len(new_features)))
            verbose = None
        return new_features

    def _filter_features(self, features):
        assert isinstance(self.drop_exact, list), "drop_exact must be a list"
        assert isinstance(self.drop_contains,
                          list), "drop_contains must be a list"
        f_keep = []
        for f in features:
            keep = True
            for contains in self.drop_contains:
                if contains in f.get_name():
                    keep = False
                    break

            if f.get_name() in self.drop_exact:
                keep = False

            if keep:
                f_keep.append(f)

        return f_keep

    def _run_dfs(self, dataframe, relationship_path, all_features, max_depth):
        """
        Create features for the provided dataframe

        Args:
            dataframe (DataFrame): Dataframe for which to create features.
            relationship_path (RelationshipPath): The path to this dataframe.
            all_features (dict[dataframe name -> dict[str -> BaseFeature]]):
                Dict containing a dict for each dataframe. Each nested dict
                has features as values with their ids as keys.
            max_depth (int) : Maximum allowed depth of features.
        """
        if max_depth is not None and max_depth < 0:
            return

        all_features[dataframe.ww.name] = {}

        """
        Step 1 - Create identity features
        """
        self._add_identity_features(all_features, dataframe)

        """
        Step 2 - Recursively build features for each dataframe in a backward relationship
        """

        backward_dataframes = self.es.get_backward_dataframes(dataframe.ww.name)
        for b_dataframe_id, sub_relationship_path in backward_dataframes:
            # Skip if we've already created features for this dataframe.
            if b_dataframe_id in all_features:
                continue

            if b_dataframe_id in self.ignore_dataframes:
                continue

            new_path = relationship_path + sub_relationship_path
            if self.allowed_paths and tuple(new_path.dataframes()) not in self.allowed_paths:
                continue

            new_max_depth = None
            if max_depth is not None:
                new_max_depth = max_depth - 1
            self._run_dfs(dataframe=self.es[b_dataframe_id],
                          relationship_path=new_path,
                          all_features=all_features,
                          max_depth=new_max_depth)

        """
        Step 3 - Create aggregation features for all deep backward relationships
        """

        backward_dataframes = self.es.get_backward_dataframes(dataframe.ww.name, deep=True)
        for b_dataframe_id, sub_relationship_path in backward_dataframes:
            if b_dataframe_id in self.ignore_dataframes:
                continue

            new_path = relationship_path + sub_relationship_path
            if self.allowed_paths and tuple(new_path.dataframes()) not in self.allowed_paths:
                continue

            self._build_agg_features(parent_dataframe=self.es[dataframe.ww.name],
                                     child_dataframe=self.es[b_dataframe_id],
                                     all_features=all_features,
                                     max_depth=max_depth,
                                     relationship_path=sub_relationship_path)

        """
        Step 4 - Create transform features of identity and aggregation features
        """

        self._build_transform_features(all_features, dataframe, max_depth=max_depth)

        """
        Step 5 - Recursively build features for each dataframe in a forward relationship
        """

        forward_dataframes = self.es.get_forward_dataframes(dataframe.ww.name)
        for f_dataframe_id, sub_relationship_path in forward_dataframes:
            # Skip if we've already created features for this dataframe.
            if f_dataframe_id in all_features:
                continue

            if f_dataframe_id in self.ignore_dataframes:
                continue

            new_path = relationship_path + sub_relationship_path
            if self.allowed_paths and tuple(new_path.dataframes()) not in self.allowed_paths:
                continue

            new_max_depth = None
            if max_depth is not None:
                new_max_depth = max_depth - 1
            self._run_dfs(dataframe=self.es[f_dataframe_id],
                          relationship_path=new_path,
                          all_features=all_features,
                          max_depth=new_max_depth)

        """
        Step 6 - Create direct features for forward relationships
        """

        forward_dataframes = self.es.get_forward_dataframes(dataframe.ww.name)
        for f_dataframe_id, sub_relationship_path in forward_dataframes:
            if f_dataframe_id in self.ignore_dataframes:
                continue

            new_path = relationship_path + sub_relationship_path
            if self.allowed_paths and tuple(new_path.dataframes()) not in self.allowed_paths:
                continue

            self._build_forward_features(
                all_features=all_features,
                relationship_path=sub_relationship_path,
                max_depth=max_depth)

        """
        Step 7 - Create transform features of direct features
        """

        self._build_transform_features(all_features, dataframe, max_depth=max_depth,
                                       require_direct_input=True)

        # now that all  features are added, build where clauses
        self._build_where_clauses(all_features, dataframe)

    def _handle_new_feature(self, new_feature, all_features):
        """Adds new feature to the dict

        Args:
            new_feature (:class:`.FeatureBase`): New feature being
                checked.
            all_features (dict[dataframe name -> dict[str -> BaseFeature]]):
                Dict containing a dict for each dataframe. Each nested dict
                has features as values with their ids as keys.

        Returns:
            dict[PrimitiveBase -> dict[feature id -> feature]]: Dict of
                features with any new features.

        Raises:
            Exception: Attempted to add a single feature multiple times
        """
        dataframe_name = new_feature.dataframe_name
        name = new_feature.unique_name()

        # Warn if this feature is already present, and it is not a seed feature.
        # It is expected that a seed feature could also be generated by dfs.
        if name in all_features[dataframe_name] and \
                name not in (f.unique_name() for f in self.seed_features):
            logger.warning('Attempting to add feature %s which is already '
                           'present. This is likely a bug.' % new_feature)
            return

        all_features[dataframe_name][name] = new_feature

    def _add_identity_features(self, all_features, dataframe):
        """converts all columns from the given dataframe into features

        Args:
            all_features (dict[dataframe name -> dict[str -> BaseFeature]]):
                Dict containing a dict for each dataframe. Each nested dict
                has features as values with their ids as keys.
            dataframe (DataFrame): DataFrame to calculate features for.
        """
        for col in dataframe.columns:
            if col in self.ignore_columns[dataframe.ww.name] or col == LTI_COLUMN_NAME:
                continue
            new_f = IdentityFeature(self.es[dataframe.ww.name].ww[col])
            self._handle_new_feature(all_features=all_features,
                                     new_feature=new_f)

        # add seed features, if any, for dfs to build on top of
        # if there are any multi output features, this will build on
        # top of each output of the feature.
        for f in self.seed_features:
            if f.dataframe_name == dataframe.ww.name:
                self._handle_new_feature(all_features=all_features,
                                         new_feature=f)

    def _build_where_clauses(self, all_features, dataframe):
        """Traverses all identity features and creates a Compare for
            each one, based on some heuristics

        Args:
            all_features (dict[dataframe name -> dict[str -> BaseFeature]]):
                Dict containing a dict for each dataframe. Each nested dict
                has features as values with their ids as keys.
          dataframe (DataFrame): DataFrame to calculate features for.
        """
        def is_valid_feature(f):
            if isinstance(f, IdentityFeature):
                return True
            if isinstance(f, DirectFeature) and getattr(f.base_features[0], "column_name", None):
                return True
            return False

        features = [f for f in all_features[dataframe.ww.name].values() if is_valid_feature(f)]
        for feat in features:
            # Get interesting_values from the EntitySet that was passed, which
            # is assumed to be the most recent version of the EntitySet.
            # Features can contain a stale EntitySet reference without
            # interesting_values
            if isinstance(feat, DirectFeature):
                df = feat.base_features[0].dataframe_name
                col = feat.base_features[0].column_name
            else:
                df = feat.dataframe_name
                col = feat.column_name
            metadata = self.es[df].ww.columns[col].metadata
            interesting_values = metadata.get('interesting_values')
            if interesting_values:
                for val in interesting_values:
                    self.where_clauses[dataframe.ww.name].add(feat == val)

    def _build_transform_features(self, all_features, dataframe, max_depth=0,
                                  require_direct_input=False):
        """Creates trans_features for all the columns in a dataframe

        Args:
            all_features (dict[dataframe name: dict->[str->:class:`BaseFeature`]]):
                Dict containing a dict for each dataframe. Each nested dict
                has features as values with their ids as keys

          dataframe (DataFrame): DataFrame to calculate features for.
        """

        new_max_depth = None
        if max_depth is not None:
            new_max_depth = max_depth - 1

        # Keep track of features to add until the end to avoid applying
        # transform primitives to features that were also built by transform primitives
        features_to_add = []

        for trans_prim in self.trans_primitives:
            current_options = self.primitive_options.get(
                trans_prim,
                self.primitive_options.get(trans_prim.name))
            if ignore_dataframe_for_primitive(current_options, dataframe):
                continue
            # if multiple input_types, only use first one for DFS
            input_types = trans_prim.input_types
            if type(input_types[0]) == list:
                input_types = input_types[0]

            matching_inputs = self._get_matching_inputs(all_features,
                                                        dataframe,
                                                        new_max_depth,
                                                        input_types,
                                                        trans_prim,
                                                        current_options,
                                                        require_direct_input=require_direct_input)

            for matching_input in matching_inputs:
                if (all(bf.number_output_features == 1 for bf in matching_input) and
                        check_transform_stacking(matching_input)):
                    new_f = TransformFeature(matching_input,
                                             primitive=trans_prim)
                    features_to_add.append(new_f)

        for groupby_prim in self.groupby_trans_primitives:
            current_options = self.primitive_options.get(
                groupby_prim,
                self.primitive_options.get(groupby_prim.name))
            if ignore_dataframe_for_primitive(current_options, dataframe, groupby=True):
                continue
            input_types = groupby_prim.input_types[:]
            # if multiple input_types, only use first one for DFS
            if type(input_types[0]) == list:
                input_types = input_types[0]
            matching_inputs = self._get_matching_inputs(all_features,
                                                        dataframe,
                                                        new_max_depth,
                                                        input_types,
                                                        groupby_prim,
                                                        current_options)

            # get columns to use as groupbys, use IDs as default unless other groupbys specified
            if any(['include_groupby_columns' in option and dataframe.ww.name in
                    option['include_groupby_columns'] for option in current_options]):
                column_schemas = 'all'
            else:
                column_schemas = [ColumnSchema(semantic_tags=['foreign_key'])]
            groupby_matches = self._features_by_type(all_features=all_features,
                                                     dataframe=dataframe,
                                                     max_depth=new_max_depth,
                                                     column_schemas=column_schemas)
            groupby_matches = filter_groupby_matches_by_options(groupby_matches, current_options)

            # If require_direct_input, require a DirectFeature in input or as a
            # groupby, and don't create features of inputs/groupbys which are
            # all direct features with the same relationship path
            for matching_input in matching_inputs:
                if (all(bf.number_output_features == 1 for bf in matching_input) and
                        check_transform_stacking(matching_input)):
                    for groupby in groupby_matches:
                        if require_direct_input and (
                            _all_direct_and_same_path(matching_input + (groupby,)) or
                            not any([isinstance(feature, DirectFeature) for
                                     feature in (matching_input + (groupby, ))])
                        ):
                            continue
                        new_f = GroupByTransformFeature(list(matching_input),
                                                        groupby=groupby[0],
                                                        primitive=groupby_prim)
                        features_to_add.append(new_f)
        for new_f in features_to_add:
            self._handle_new_feature(all_features=all_features,
                                     new_feature=new_f)

    def _build_forward_features(self, all_features, relationship_path, max_depth=0):
        _, relationship = relationship_path[0]

        child_dataframe_name = relationship.child_dataframe.ww.name
        parent_dataframe = relationship.parent_dataframe

        features = self._features_by_type(
            all_features=all_features,
            dataframe=parent_dataframe,
            max_depth=max_depth,
            column_schemas='all')

        for f in features:
            if self._feature_in_relationship_path(relationship_path, f):
                continue

            # limits allowing direct features of agg_feats with where clauses
            if isinstance(f, AggregationFeature):
                deep_base_features = [f] + f.get_dependencies(deep=True)
                for feat in deep_base_features:
                    if isinstance(feat, AggregationFeature) and feat.where is not None:
                        continue

            new_f = DirectFeature(f, child_dataframe_name, relationship=relationship)

            self._handle_new_feature(all_features=all_features,
                                     new_feature=new_f)

    def _build_agg_features(self, all_features, parent_dataframe, child_dataframe,
                            max_depth, relationship_path):
        new_max_depth = None
        if max_depth is not None:
            new_max_depth = max_depth - 1
        for agg_prim in self.agg_primitives:
            current_options = self.primitive_options.get(
                agg_prim,
                self.primitive_options.get(agg_prim.name))

            if ignore_dataframe_for_primitive(current_options, child_dataframe):
                continue
            # if multiple input_types, only use first one for DFS
            input_types = agg_prim.input_types
            if type(input_types[0]) == list:
                input_types = input_types[0]

            def feature_filter(f):
                # Remove direct features of parent dataframe and features in relationship path.
                return (not _direct_of_dataframe(f, parent_dataframe)) \
                    and not self._feature_in_relationship_path(relationship_path, f)

            matching_inputs = self._get_matching_inputs(all_features,
                                                        child_dataframe,
                                                        new_max_depth,
                                                        input_types,
                                                        agg_prim,
                                                        current_options,
                                                        feature_filter=feature_filter)

            matching_inputs = filter_matches_by_options(matching_inputs,
                                                        current_options)
            wheres = list(self.where_clauses[child_dataframe.ww.name])

            for matching_input in matching_inputs:
                if not check_stacking(agg_prim, matching_input):
                    continue
                new_f = AggregationFeature(matching_input,
                                           parent_dataframe_name=parent_dataframe.ww.name,
                                           relationship_path=relationship_path,
                                           primitive=agg_prim)

                self._handle_new_feature(new_f, all_features)

                # limit the stacking of where features
                # count up the the number of where features
                # in this feature and its dependencies
                feat_wheres = []
                for f in matching_input:
                    if isinstance(f, AggregationFeature) and f.where is not None:
                        feat_wheres.append(f)
                    for feat in f.get_dependencies(deep=True):
                        if (isinstance(feat, AggregationFeature) and
                                feat.where is not None):
                            feat_wheres.append(feat)

                if len(feat_wheres) >= self.where_stacking_limit:
                    continue

                # limits the aggregation feature by the given allowed feature types.
                if not any([issubclass(type(agg_prim), type(primitive))
                            for primitive in self.where_primitives]):
                    continue

                for where in wheres:
                    # limits the where feats so they are different than base feats
                    base_names = [f.unique_name() for f in new_f.base_features]
                    if any([base_feat.unique_name() in base_names for base_feat in where.base_features]):
                        continue

                    new_f = AggregationFeature(matching_input,
                                               parent_dataframe_name=parent_dataframe.ww.name,
                                               relationship_path=relationship_path,
                                               where=where,
                                               primitive=agg_prim)
                    self._handle_new_feature(new_f, all_features)

    def _features_by_type(self, all_features, dataframe, max_depth,
                          column_schemas=None):

        selected_features = []

        if max_depth is not None and max_depth < 0:
            return selected_features

        if dataframe.ww.name not in all_features:
            return selected_features

        dataframe_features = all_features[dataframe.ww.name].copy()
        for fname, feature in all_features[dataframe.ww.name].items():
            outputs = feature.number_output_features
            if outputs > 1:
                del(dataframe_features[fname])
                for i in range(outputs):
                    new_feat = feature[i]
                    dataframe_features[new_feat.unique_name()] = new_feat

        for feat in dataframe_features:
            f = dataframe_features[feat]
            if (column_schemas == 'all' or
                    any(is_valid_input(f.column_schema, schema) for schema in column_schemas)):
                if max_depth is None or f.get_depth(stop_at=self.seed_features) <= max_depth:
                    selected_features.append(f)

        return selected_features

    def _feature_in_relationship_path(self, relationship_path, feature):
        # must be identity feature to be in the relationship path
        if not isinstance(feature, IdentityFeature):
            return False

        for _, relationship in relationship_path:
            if relationship.child_name == feature.dataframe_name and \
               relationship._child_column_name == feature.column_name:
                return True

            if relationship.parent_name == feature.dataframe_name and \
               relationship._parent_column_name == feature.column_name:
                return True

        return False

    def _get_matching_inputs(self, all_features, dataframe, max_depth, input_types,
                             primitive, primitive_options, require_direct_input=False,
                             feature_filter=None):

        features = self._features_by_type(all_features=all_features,
                                          dataframe=dataframe,
                                          max_depth=max_depth,
                                          column_schemas=list(input_types))
        if feature_filter:
            features = [f for f in features if feature_filter(f)]

        matching_inputs = match(input_types, features,
                                commutative=primitive.commutative,
                                require_direct_input=require_direct_input)

        if require_direct_input:
            # Don't create trans features of inputs which are all direct
            # features with the same relationship_path.
            matching_inputs = {inputs for inputs in matching_inputs
                               if not _all_direct_and_same_path(inputs)}
        matching_inputs = filter_matches_by_options(matching_inputs,
                                                    primitive_options,
                                                    commutative=primitive.commutative)

        # Don't build features on numeric foreign key columns
        matching_inputs = [match for match in matching_inputs if not _match_contains_numeric_foreign_key(match)]

        return matching_inputs


def _match_contains_numeric_foreign_key(match):
    match_schema = ColumnSchema(semantic_tags={'foreign_key', 'numeric'})
    return any(is_valid_input(f.column_schema, match_schema) for f in match)


def check_transform_stacking(inputs):
    # Avoid transform stacking when building from direct features
    for f in inputs:
        if isinstance(f.primitive, TransformPrimitive):
            return False
        if isinstance(f, DirectFeature) and isinstance(f.base_features[0].primitive, TransformPrimitive):
            return False
    return True


def check_stacking(primitive, inputs):
    """checks if features in inputs can be used with supplied primitive
       using the stacking rules"""
    if primitive.stack_on_self is False:
        for f in inputs:
            if isinstance(f.primitive, primitive.__class__):
                return False

    if primitive.stack_on_exclude is not None:
        for f in inputs:
            if isinstance(f.primitive, tuple(primitive.stack_on_exclude)):
                return False

    # R TODO: handle this
    for f in inputs:
        if f.number_output_features > 1:
            return False

    for f in inputs:
        if f.primitive.base_of_exclude is not None:
            if isinstance(primitive, tuple(f.primitive.base_of_exclude)):
                return False

    for f in inputs:
        if primitive.stack_on_self is True:
            if isinstance(f.primitive, primitive.__class__):
                continue
        if primitive.stack_on is not None:
            if isinstance(f.primitive, tuple(primitive.stack_on)):
                continue
        else:
            continue
        if f.primitive.base_of is not None:
            if primitive.__class__ in f.primitive.base_of:
                continue
        else:
            continue
        return False

    return True


def match_by_schema(features, column_schema):
    matches = []
    for f in features:
        if is_valid_input(f.column_schema, column_schema):
            matches += [f]
    return matches


def match(input_types, features, replace=False, commutative=False, require_direct_input=False):
    to_match = input_types[0]

    matches = match_by_schema(features, to_match)

    if len(input_types) == 1:
        return [(m,) for m in matches
                if (not require_direct_input or isinstance(m, DirectFeature))]

    matching_inputs = set([])

    for m in matches:
        copy = features[:]

        if not replace:
            copy = [c for c in copy if c.unique_name() != m.unique_name()]

        # If we need a DirectFeature and this is not a DirectFeature then one of the rest must be.
        still_require_direct_input = require_direct_input and not isinstance(m, DirectFeature)
        rest = match(input_types[1:], copy, replace,
                     require_direct_input=still_require_direct_input)

        for r in rest:
            new_match = [m] + list(r)

            # commutative uses frozenset instead of tuple because it doesn't
            # want multiple orderings of the same input
            if commutative:
                new_match = frozenset(new_match)
            else:
                new_match = tuple(new_match)
            matching_inputs.add(new_match)

    if commutative:
        matching_inputs = {tuple(sorted(s, key=lambda x: x.get_name().lower())) for s in matching_inputs}

    return matching_inputs


def handle_primitive(primitive):
    if not isinstance(primitive, PrimitiveBase):
        primitive = primitive()
    assert isinstance(primitive, PrimitiveBase), "must be a primitive"
    return primitive


def check_primitive(primitive, prim_type):
    if prim_type == "transform" or prim_type == "groupby transform":
        prim_dict = primitives.get_transform_primitives()
        supertype = TransformPrimitive
        arg_name = "trans_primitives" if prim_type == "transform" else "groupby_trans_primitives"
        s = "a transform"
    if prim_type == "aggregation" or prim_type == "where":
        prim_dict = primitives.get_aggregation_primitives()
        supertype = AggregationPrimitive
        arg_name = "agg_primitives" if prim_type == "aggregation" else "where_primitives"
        s = "an aggregation"

    if isinstance(primitive, str):
        prim_string = camel_and_title_to_snake(primitive)
        if prim_string not in prim_dict:
            raise ValueError("Unknown {} primitive {}. "
                             "Call ft.primitives.list_primitives() to get"
                             " a list of available primitives".format(prim_type, primitive))
        primitive = prim_dict[prim_string]
    primitive = handle_primitive(primitive)
    if not isinstance(primitive, supertype):
        raise ValueError("Primitive {} in {} is not {} "
                         "primitive".format(type(primitive), arg_name, s))
    return primitive


def _all_direct_and_same_path(input_features):
    return all(isinstance(f, DirectFeature) for f in input_features) and \
        _features_have_same_path(input_features)


def _features_have_same_path(input_features):
    path = input_features[0].relationship_path

    for f in input_features[1:]:
        if f.relationship_path != path:
            return False

    return True


def _direct_of_dataframe(feature, parent_dataframe):
    return isinstance(feature, DirectFeature) \
        and feature.parent_dataframe_name == parent_dataframe.ww.name
