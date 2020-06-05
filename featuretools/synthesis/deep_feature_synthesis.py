import logging
from collections import defaultdict

from dask import dataframe as dd

from featuretools import primitives, variable_types
from featuretools.entityset.relationship import RelationshipPath
from featuretools.feature_base import (
    AggregationFeature,
    DirectFeature,
    GroupByTransformFeature,
    IdentityFeature,
    TransformFeature
)
from featuretools.primitives.base import (
    AggregationPrimitive,
    PrimitiveBase,
    TransformPrimitive
)
from featuretools.primitives.options_utils import (
    filter_groupby_matches_by_options,
    filter_matches_by_options,
    generate_all_primitive_options,
    ignore_entity_for_primitive
)
from featuretools.variable_types import Boolean, Discrete, Id, Numeric

logger = logging.getLogger('featuretools')


class DeepFeatureSynthesis(object):
    """Automatically produce features for a target entity in an Entityset.

        Args:
            target_entity_id (str): Id of entity for which to build features.

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

            allowed_paths (list[list[str]], optional): Allowed entity paths to make
                features for. If None, use all paths.

            ignore_entities (list[str], optional): List of entities to
                blacklist when creating features. If None, use all entities.

            ignore_variables (dict[str -> list[str]], optional): List of specific
                variables within each entity to blacklist when creating features.
                If None, use all variables.

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


                ``"include_entities"``
                    List of entities to be included when creating features for
                    the primitive(s). All other entities will be ignored
                    (list[str]).
                ``"ignore_entities"``
                    List of entities to be blacklisted when creating features
                    for the primitive(s) (list[str]).
                ``"include_variables"``
                    List of specific variables within each entity to include when
                    creating feautres for the primitive(s). All other variables
                    in a given entity will be ignored (dict[str -> list[str]]).
                ``"ignore_variables"``
                    List of specific variables within each entityt to blacklist
                    when creating features for the primitive(s) (dict[str ->
                    list[str]]).
                ``"include_groupby_entities"``
                    List of Entities to be included when finding groupbys. All
                    other entities will be ignored (list[str]).
                ``"ignore_groupby_entities"``
                    List of entities to blacklist when finding groupbys
                    (list[str]).
                ``"include_groupby_variables"``
                    List of specific variables within each entity to include as
                    groupbys, if applicable. All other variables in each
                    entity will be ignored (dict[str -> list[str]]).
                ``"ignore_groupby_variables"``
                    List of specific variables within each entity to blacklist
                    as groupbys (dict[str -> list[str]]).
        """

    def __init__(self,
                 target_entity_id,
                 entityset,
                 agg_primitives=None,
                 trans_primitives=None,
                 where_primitives=None,
                 groupby_trans_primitives=None,
                 max_depth=2,
                 max_features=-1,
                 allowed_paths=None,
                 ignore_entities=None,
                 ignore_variables=None,
                 primitive_options=None,
                 seed_features=None,
                 drop_contains=None,
                 drop_exact=None,
                 where_stacking_limit=1):

        if target_entity_id not in entityset.entity_dict:
            es_name = entityset.id or 'entity set'
            msg = 'Provided target entity %s does not exist in %s' % (target_entity_id, es_name)
            raise KeyError(msg)

        # need to change max_depth to None because DFs terminates when  <0
        if max_depth == -1:
            max_depth = None
        self.max_depth = max_depth

        self.max_features = max_features

        self.allowed_paths = allowed_paths
        if self.allowed_paths:
            self.allowed_paths = set()
            for path in allowed_paths:
                self.allowed_paths.add(tuple(path))

        if ignore_entities is None:
            self.ignore_entities = set()
        else:
            if not isinstance(ignore_entities, list):
                raise TypeError('ignore_entities must be a list')
            assert target_entity_id not in ignore_entities,\
                "Can't ignore target_entity!"
            self.ignore_entities = set(ignore_entities)

        self.ignore_variables = defaultdict(set)
        if ignore_variables is not None:
            # check if ignore_variables is not {str: list}
            if not all(isinstance(i, str) for i in ignore_variables.keys()) or not all(isinstance(i, list) for i in ignore_variables.values()):
                raise TypeError('ignore_variables should be dict[str -> list]')
            # check if list values are all of type str
            elif not all(all(isinstance(v, str) for v in value) for value in ignore_variables.values()):
                raise TypeError('list values should be of type str')
            for eid, vars in ignore_variables.items():
                self.ignore_variables[eid] = set(vars)
        self.target_entity_id = target_entity_id
        self.es = entityset

        if agg_primitives is None:
            agg_primitives = primitives.get_default_aggregation_primitives()
            if any(isinstance(e.df, dd.DataFrame) for e in self.es.entities):
                agg_primitives = [p for p in agg_primitives if p.dask_compatible]
        self.agg_primitives = []
        agg_prim_dict = primitives.get_aggregation_primitives()
        for a in agg_primitives:
            if isinstance(a, str):
                if a.lower() not in agg_prim_dict:
                    raise ValueError("Unknown aggregation primitive {}. ".format(a),
                                     "Call ft.primitives.list_primitives() to get",
                                     " a list of available primitives")
                a = agg_prim_dict[a.lower()]
            a = handle_primitive(a)
            if not isinstance(a, AggregationPrimitive):
                raise ValueError("Primitive {} in agg_primitives is not an "
                                 "aggregation primitive".format(type(a)))
            self.agg_primitives.append(a)

        if trans_primitives is None:
            trans_primitives = primitives.get_default_transform_primitives()
            if any(isinstance(e.df, dd.DataFrame) for e in self.es.entities):
                trans_primitives = [p for p in trans_primitives if p.dask_compatible]
        self.trans_primitives = []
        for t in trans_primitives:
            t = check_trans_primitive(t)
            self.trans_primitives.append(t)

        if where_primitives is None:
            where_primitives = [primitives.Count]
        self.where_primitives = []
        for p in where_primitives:
            if isinstance(p, str):
                prim_obj = agg_prim_dict.get(p.lower(), None)
                if prim_obj is None:
                    raise ValueError("Unknown where primitive {}. ".format(p),
                                     "Call ft.primitives.list_primitives() to get",
                                     " a list of available primitives")
                p = prim_obj
            p = handle_primitive(p)
            self.where_primitives.append(p)

        if groupby_trans_primitives is None:
            groupby_trans_primitives = []
        self.groupby_trans_primitives = []
        for p in groupby_trans_primitives:
            p = check_trans_primitive(p)
            self.groupby_trans_primitives.append(p)

        if primitive_options is None:
            primitive_options = {}
        all_primitives = self.trans_primitives + self.agg_primitives + \
            self.where_primitives + self.groupby_trans_primitives
        if any(isinstance(entity.df, dd.DataFrame) for entity in self.es.entities):
            if not all([primitive.dask_compatible for primitive in all_primitives]):
                bad_primitives = ", ".join([prim.name for prim in all_primitives if not prim.dask_compatible])
                raise ValueError('Selected primitives are incompatible with Dask EntitySets: {}'.format(bad_primitives))

        self.primitive_options, self.ignore_entities, self.ignore_variables =\
            generate_all_primitive_options(all_primitives,
                                           primitive_options,
                                           self.ignore_entities,
                                           self.ignore_variables,
                                           self.es)
        self.seed_features = seed_features or []
        self.drop_exact = drop_exact or []
        self.drop_contains = drop_contains or []
        self.where_stacking_limit = where_stacking_limit

    def build_features(self, return_variable_types=None, verbose=False):
        """Automatically builds feature definitions for target
            entity using Deep Feature Synthesis algorithm

        Args:
            return_variable_types (list[Variable] or str, optional): Types of
                variables to return. If None, default to
                Numeric, Discrete, and Boolean. If given as
                the string 'all', use all available variable types.

            verbose (bool, optional): If True, print progress.

        Returns:
            list[BaseFeature]: Returns a list of
                features for target entity, sorted by feature depth
                (shallow first).
        """
        all_features = {}

        self.where_clauses = defaultdict(set)

        if return_variable_types is None:
            return_variable_types = [Numeric, Discrete, Boolean]
        elif return_variable_types == 'all':
            pass
        else:
            msg = "return_variable_types must be a list, or 'all'"
            assert isinstance(return_variable_types, list), msg

        self._run_dfs(self.es[self.target_entity_id], RelationshipPath([]),
                      all_features, max_depth=self.max_depth)

        new_features = list(all_features[self.target_entity_id].values())

        def filt(f):
            # remove identity features of the ID field of the target entity
            if (isinstance(f, IdentityFeature) and
                    f.entity.id == self.target_entity_id and
                    f.variable.id == self.es[self.target_entity_id].index):
                return False

            return True

        # filter out features with undesired return types
        if return_variable_types != 'all':
            new_features = [
                f for f in new_features
                if any(issubclass(
                    f.variable_type, vt) for vt in return_variable_types)]

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

    def _run_dfs(self, entity, relationship_path, all_features, max_depth):
        """
        Create features for the provided entity

        Args:
            entity (Entity): Entity for which to create features.
            relationship_path (RelationshipPath): The path to this entity.
            all_features (dict[Entity.id -> dict[str -> BaseFeature]]):
                Dict containing a dict for each entity. Each nested dict
                has features as values with their ids as keys.
            max_depth (int) : Maximum allowed depth of features.
        """
        if max_depth is not None and max_depth < 0:
            return

        all_features[entity.id] = {}

        """
        Step 1 - Create identity features
        """
        self._add_identity_features(all_features, entity)

        """
        Step 2 - Recursively build features for each entity in a backward relationship
        """

        backward_entities = self.es.get_backward_entities(entity.id)
        for b_entity_id, sub_relationship_path in backward_entities:
            # Skip if we've already created features for this entity.
            if b_entity_id in all_features:
                continue

            if b_entity_id in self.ignore_entities:
                continue

            new_path = relationship_path + sub_relationship_path
            if self.allowed_paths and tuple(new_path.entities()) not in self.allowed_paths:
                continue

            new_max_depth = None
            if max_depth is not None:
                new_max_depth = max_depth - 1
            self._run_dfs(entity=self.es[b_entity_id],
                          relationship_path=new_path,
                          all_features=all_features,
                          max_depth=new_max_depth)

        """
        Step 3 - Create aggregation features for all deep backward relationships
        """

        backward_entities = self.es.get_backward_entities(entity.id, deep=True)
        for b_entity_id, sub_relationship_path in backward_entities:
            if b_entity_id in self.ignore_entities:
                continue

            new_path = relationship_path + sub_relationship_path
            if self.allowed_paths and tuple(new_path.entities()) not in self.allowed_paths:
                continue

            self._build_agg_features(parent_entity=self.es[entity.id],
                                     child_entity=self.es[b_entity_id],
                                     all_features=all_features,
                                     max_depth=max_depth,
                                     relationship_path=sub_relationship_path)

        """
        Step 4 - Create transform features of identity and aggregation features
        """

        self._build_transform_features(all_features, entity, max_depth=max_depth)

        """
        Step 5 - Recursively build features for each entity in a forward relationship
        """

        forward_entities = self.es.get_forward_entities(entity.id)
        for f_entity_id, sub_relationship_path in forward_entities:
            # Skip if we've already created features for this entity.
            if f_entity_id in all_features:
                continue

            if f_entity_id in self.ignore_entities:
                continue

            new_path = relationship_path + sub_relationship_path
            if self.allowed_paths and tuple(new_path.entities()) not in self.allowed_paths:
                continue

            new_max_depth = None
            if max_depth is not None:
                new_max_depth = max_depth - 1
            self._run_dfs(entity=self.es[f_entity_id],
                          relationship_path=new_path,
                          all_features=all_features,
                          max_depth=new_max_depth)

        """
        Step 6 - Create direct features for forward relationships
        """

        forward_entities = self.es.get_forward_entities(entity.id)
        for f_entity_id, sub_relationship_path in forward_entities:
            if f_entity_id in self.ignore_entities:
                continue

            new_path = relationship_path + sub_relationship_path
            if self.allowed_paths and tuple(new_path.entities()) not in self.allowed_paths:
                continue

            self._build_forward_features(
                all_features=all_features,
                relationship_path=sub_relationship_path,
                max_depth=max_depth)

        """
        Step 7 - Create transform features of direct features
        """
        self._build_transform_features(all_features, entity, max_depth=max_depth,
                                       require_direct_input=True)

        # now that all  features are added, build where clauses
        self._build_where_clauses(all_features, entity)

    def _handle_new_feature(self, new_feature, all_features):
        """Adds new feature to the dict

        Args:
            new_feature (:class:`.FeatureBase`): New feature being
                checked.
            all_features (dict[Entity.id -> dict[str -> BaseFeature]]):
                Dict containing a dict for each entity. Each nested dict
                has features as values with their ids as keys.

        Returns:
            dict[PrimitiveBase -> dict[featureid -> feature]]: Dict of
                features with any new features.

        Raises:
            Exception: Attempted to add a single feature multiple times
        """
        entity_id = new_feature.entity.id
        name = new_feature.unique_name()

        # Warn if this feature is already present, and it is not a seed feature.
        # It is expected that a seed feature could also be generated by dfs.
        if name in all_features[entity_id] and \
                name not in (f.unique_name() for f in self.seed_features):
            logger.warning('Attempting to add feature %s which is already '
                           'present. This is likely a bug.' % new_feature)
            return

        all_features[entity_id][name] = new_feature

    def _add_identity_features(self, all_features, entity):
        """converts all variables from the given entity into features

        Args:
            all_features (dict[Entity.id -> dict[str -> BaseFeature]]):
                Dict containing a dict for each entity. Each nested dict
                has features as values with their ids as keys.
            entity (Entity): Entity to calculate features for.
        """
        variables = entity.variables
        for v in variables:
            if v.name in self.ignore_variables[entity.id]:
                continue
            new_f = IdentityFeature(variable=v)
            self._handle_new_feature(all_features=all_features,
                                     new_feature=new_f)

        # add seed features, if any, for dfs to build on top of
        # if there are any multi output features, this will build on
        # top of each output of the feature.
        for f in self.seed_features:
            if f.entity.id == entity.id:
                self._handle_new_feature(all_features=all_features,
                                         new_feature=f)

    def _build_where_clauses(self, all_features, entity):
        """Traverses all identity features and creates a Compare for
            each one, based on some heuristics

        Args:
            all_features (dict[Entity.id -> dict[str -> BaseFeature]]):
                Dict containing a dict for each entity. Each nested dict
                has features as values with their ids as keys.
          entity (Entity): Entity to calculate features for.
        """
        features = [f for f in all_features[entity.id].values()
                    if getattr(f, "variable", None)]

        for feat in features:
            # Get interesting_values from the EntitySet that was passed, which
            # is assumed to be the most recent version of the EntitySet.
            # Features can contain a stale EntitySet reference without
            # interesting_values
            variable = self.es[feat.variable.entity.id][feat.variable.id]
            if variable.interesting_values.empty:
                continue

            for val in variable.interesting_values:
                self.where_clauses[entity.id].add(feat == val)

    def _build_transform_features(self, all_features, entity, max_depth=0,
                                  require_direct_input=False):
        """Creates trans_features for all the variables in an entity

        Args:
            all_features (dict[:class:`.Entity`.id:dict->[str->:class:`BaseFeature`]]):
                Dict containing a dict for each entity. Each nested dict
                has features as values with their ids as keys

          entity (Entity): Entity to calculate features for.
        """
        new_max_depth = None
        if max_depth is not None:
            new_max_depth = max_depth - 1

        for trans_prim in self.trans_primitives:
            current_options = self.primitive_options.get(
                trans_prim,
                self.primitive_options.get(trans_prim.name))
            if ignore_entity_for_primitive(current_options, entity):
                continue
            # if multiple input_types, only use first one for DFS
            input_types = trans_prim.input_types
            if type(input_types[0]) == list:
                input_types = input_types[0]

            matching_inputs = self._get_matching_inputs(all_features,
                                                        entity,
                                                        new_max_depth,
                                                        input_types,
                                                        trans_prim,
                                                        current_options,
                                                        require_direct_input=require_direct_input)

            for matching_input in matching_inputs:
                if all(bf.number_output_features == 1 for bf in matching_input):
                    new_f = TransformFeature(matching_input,
                                             primitive=trans_prim)
                    self._handle_new_feature(all_features=all_features,
                                             new_feature=new_f)

        for groupby_prim in self.groupby_trans_primitives:
            current_options = self.primitive_options.get(
                groupby_prim,
                self.primitive_options.get(groupby_prim.name))
            if ignore_entity_for_primitive(current_options, entity, groupby=True):
                continue
            input_types = groupby_prim.input_types[:]
            # if multiple input_types, only use first one for DFS
            if type(input_types[0]) == list:
                input_types = input_types[0]
            matching_inputs = self._get_matching_inputs(all_features,
                                                        entity,
                                                        new_max_depth,
                                                        input_types,
                                                        groupby_prim,
                                                        current_options)

            # get columns to use as groupbys, use IDs as default unless other groupbys specified
            if any(['include_groupby_variables' in option and entity.id in
                    option['include_groupby_variables'] for option in current_options]):
                default_type = variable_types.PandasTypes._all
            else:
                default_type = set([Id])
            groupby_matches = self._features_by_type(all_features=all_features,
                                                     entity=entity,
                                                     max_depth=new_max_depth,
                                                     variable_type=default_type)
            groupby_matches = filter_groupby_matches_by_options(groupby_matches, current_options)

            # If require_direct_input, require a DirectFeature in input or as a
            # groupby, and don't create features of inputs/groupbys which are
            # all direct features with the same relationship path
            for matching_input in matching_inputs:
                if all(bf.number_output_features == 1 for bf in matching_input):
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
                        self._handle_new_feature(all_features=all_features,
                                                 new_feature=new_f)

    def _build_forward_features(self, all_features, relationship_path, max_depth=0):
        _, relationship = relationship_path[0]
        child_entity = relationship.child_entity
        parent_entity = relationship.parent_entity

        features = self._features_by_type(
            all_features=all_features,
            entity=parent_entity,
            max_depth=max_depth,
            variable_type=variable_types.PandasTypes._all)

        for f in features:
            if self._feature_in_relationship_path(relationship_path, f):
                continue

            # limits allowing direct features of agg_feats with where clauses
            if isinstance(f, AggregationFeature):
                deep_base_features = [f] + f.get_dependencies(deep=True)
                for feat in deep_base_features:
                    if isinstance(feat, AggregationFeature) and feat.where is not None:
                        continue

            new_f = DirectFeature(f, child_entity, relationship=relationship)

            self._handle_new_feature(all_features=all_features,
                                     new_feature=new_f)

    def _build_agg_features(self, all_features, parent_entity, child_entity,
                            max_depth, relationship_path):
        new_max_depth = None
        if max_depth is not None:
            new_max_depth = max_depth - 1
        for agg_prim in self.agg_primitives:
            current_options = self.primitive_options.get(
                agg_prim,
                self.primitive_options.get(agg_prim.name))

            if ignore_entity_for_primitive(current_options, child_entity):
                continue
            # if multiple input_types, only use first one for DFS
            input_types = agg_prim.input_types
            if type(input_types[0]) == list:
                input_types = input_types[0]

            def feature_filter(f):
                # Remove direct features of parent entity and features in relationship path.
                return (not _direct_of_entity(f, parent_entity)) \
                    and not self._feature_in_relationship_path(relationship_path, f)

            matching_inputs = self._get_matching_inputs(all_features,
                                                        child_entity,
                                                        new_max_depth,
                                                        input_types,
                                                        agg_prim,
                                                        current_options,
                                                        feature_filter=feature_filter)
            matching_inputs = filter_matches_by_options(matching_inputs,
                                                        current_options)
            wheres = list(self.where_clauses[child_entity.id])

            for matching_input in matching_inputs:
                if not check_stacking(agg_prim, matching_input):
                    continue
                new_f = AggregationFeature(matching_input,
                                           parent_entity=parent_entity,
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
                                               parent_entity=parent_entity,
                                               relationship_path=relationship_path,
                                               where=where,
                                               primitive=agg_prim)

                    self._handle_new_feature(new_f, all_features)

    def _features_by_type(self, all_features, entity, max_depth,
                          variable_type=None):

        selected_features = []

        if max_depth is not None and max_depth < 0:
            return selected_features

        if entity.id not in all_features:
            return selected_features

        entity_features = all_features[entity.id].copy()
        for fname, feature in all_features[entity.id].items():
            outputs = feature.number_output_features
            if outputs > 1:
                del(entity_features[fname])
                for i in range(outputs):
                    new_feat = feature[i]
                    entity_features[new_feat.unique_name()] = new_feat

        for feat in entity_features:
            f = entity_features[feat]
            if (variable_type == variable_types.PandasTypes._all or
                    f.variable_type == variable_type or
                    any(issubclass(f.variable_type, vt) for vt in variable_type)):
                if max_depth is None or f.get_depth(stop_at=self.seed_features) <= max_depth:
                    selected_features.append(f)

        return selected_features

    def _feature_in_relationship_path(self, relationship_path, feature):
        # must be identity feature to be in the relationship path
        if not isinstance(feature, IdentityFeature):
            return False

        for _, relationship in relationship_path:
            if relationship.child_entity.id == feature.entity.id and \
               relationship.child_variable.id == feature.variable.id:
                return True

            if relationship.parent_entity.id == feature.entity.id and \
               relationship.parent_variable.id == feature.variable.id:
                return True

        return False

    def _get_matching_inputs(self, all_features, entity, max_depth, input_types,
                             primitive, primitive_options, require_direct_input=False,
                             feature_filter=None):
        features = self._features_by_type(all_features=all_features,
                                          entity=entity,
                                          max_depth=max_depth,
                                          variable_type=set(input_types))
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
        matching_inputs = filter_matches_by_options(matching_inputs, primitive_options)
        return matching_inputs


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
            if primitive in f.base_of_exclude:
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


def match_by_type(features, t):
    matches = []
    for f in features:
        if issubclass(f.variable_type, t):
            matches += [f]
    return matches


def match(input_types, features, replace=False, commutative=False, require_direct_input=False):
    to_match = input_types[0]
    matches = match_by_type(features, to_match)

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


def check_trans_primitive(primitive):
    trans_prim_dict = primitives.get_transform_primitives()

    if isinstance(primitive, str):
        if primitive.lower() not in trans_prim_dict:
            raise ValueError("Unknown transform primitive {}. ".format(primitive),
                             "Call ft.primitives.list_primitives() to get",
                             " a list of available primitives")
        primitive = trans_prim_dict[primitive.lower()]
    primitive = handle_primitive(primitive)
    if not isinstance(primitive, TransformPrimitive):
        raise ValueError("Primitive {} in trans_primitives or "
                         "groupby_trans_primitives is not a transform "
                         "primitive".format(type(primitive)))
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


def _direct_of_entity(feature, parent_entity):
    return isinstance(feature, DirectFeature) \
        and feature.parent_entity.id == parent_entity.id
