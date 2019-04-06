import logging
from builtins import filter, object, str
from collections import defaultdict

from featuretools import primitives, variable_types
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
from featuretools.utils import is_string
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

            max_hlevel (int, optional) :  #TODO how to document.
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
        """

    def __init__(self,
                 target_entity_id,
                 entityset,
                 agg_primitives=None,
                 trans_primitives=None,
                 where_primitives=None,
                 groupby_trans_primitives=None,
                 max_depth=2,
                 max_hlevel=2,
                 max_features=-1,
                 allowed_paths=None,
                 ignore_entities=None,
                 ignore_variables=None,
                 seed_features=None,
                 drop_contains=None,
                 drop_exact=None,
                 where_stacking_limit=1):
        # need to change max_depth and max_hlevel to None because DFs terminates when  <0
        if max_depth == -1:
            max_depth = None
        self.max_depth = max_depth

        if max_hlevel == -1:
            max_hlevel = None
        self.max_hlevel = max_hlevel

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
            for eid, vars in ignore_variables.items():
                self.ignore_variables[eid] = set(vars)
        self.target_entity_id = target_entity_id
        self.es = entityset

        if agg_primitives is None:
            agg_primitives = [primitives.Sum, primitives.Std, primitives.Max, primitives.Skew,
                              primitives.Min, primitives.Mean, primitives.Count,
                              primitives.PercentTrue, primitives.NUnique, primitives.Mode]
        self.agg_primitives = []
        agg_prim_dict = primitives.get_aggregation_primitives()
        for a in agg_primitives:
            if is_string(a):
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
            trans_primitives = [primitives.Day, primitives.Year, primitives.Month,
                                primitives.Weekday, primitives.Haversine,
                                primitives.NumWords, primitives.NumCharacters]  # primitives.TimeSince
        self.trans_primitives = []
        for t in trans_primitives:
            t = check_trans_primitive(t)
            self.trans_primitives.append(t)

        if where_primitives is None:
            where_primitives = [primitives.Count]
        self.where_primitives = []
        for p in where_primitives:
            if is_string(p):
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
        for e in self.es.entities:
            if e not in self.ignore_entities:
                all_features[e.id] = {}

        self.where_clauses = defaultdict(set)

        if return_variable_types is None:
            return_variable_types = [Numeric, Discrete, Boolean]
        elif return_variable_types == 'all':
            pass
        else:
            msg = "return_variable_types must be a list, or 'all'"
            assert isinstance(return_variable_types, list), msg

        self._run_dfs(self.es[self.target_entity_id], [],
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

        # sanity check for duplicate features
        hashes = [f.hash() for f in new_features]
        assert len(set([f for f in hashes if hashes.count(f) > 1])) == 0, \
            'Multiple features with same name' + \
            str(set([f for f in hashes if hashes.count(f) > 1]))

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

    def _run_dfs(self, entity, entity_path, all_features, max_depth):
        """
        create features for the provided entity

        Args:
            entity (Entity): Entity for which to create features.
            entity_path (list[str]): List of entity ids.
            all_features (dict[Entity.id -> dict[str -> BaseFeature]]):
                Dict containing a dict for each entity. Each nested dict
                has features as values with their ids as keys.
            max_depth (int) : Maximum allowed depth of features.
        """
        if max_depth is not None and max_depth < 0:
            return

        entity_path.append(entity.id)
        """
        Step 1 - Recursively build features for each entity in a backward relationship
        """

        backward_entities = self.es.get_backward_entities(entity.id)
        backward_entities = [b_id for b_id in backward_entities
                             if b_id not in self.ignore_entities]
        for b_entity_id in backward_entities:
            # if in path, we've already built features
            if b_entity_id in entity_path:
                continue

            if self.allowed_paths and tuple(entity_path + [b_entity_id]) not in self.allowed_paths:
                continue
            new_max_depth = None
            if max_depth is not None:
                new_max_depth = max_depth - 1
            self._run_dfs(entity=self.es[b_entity_id],
                          entity_path=list(entity_path),
                          all_features=all_features,
                          max_depth=new_max_depth)

        """
        Step 2 - Create agg_feat features for all deep backward relationships
        """

        # search for deep relationships of that passed the traversal filter
        build_entities = list(backward_entities)
        for b_id in backward_entities:
            build_entities += self.es.get_backward_entities(b_id, deep=True)

        for b_entity_id in build_entities:
            if b_entity_id in self.ignore_entities:
                continue
            if self.allowed_paths and tuple(entity_path + [b_entity_id]) not in self.allowed_paths:
                continue
            self._build_agg_features(parent_entity=self.es[entity.id],
                                     child_entity=self.es[b_entity_id],
                                     all_features=all_features,
                                     max_depth=max_depth)

        """
        Step 3 - Create Transform features
        """
        self._build_transform_features(
            all_features, entity, max_depth=max_depth)

        """
        Step 4 - Recursively build features for each entity in a forward relationship
        """
        forward_entities = self.es.get_forward_entities(entity.id)
        forward_entities = [f_id for f_id in forward_entities
                            if f_id not in self.ignore_entities]

        for f_entity_id in forward_entities:
            # if in path, we've already built features
            if f_entity_id in entity_path:
                continue

            if self.allowed_paths and tuple(entity_path + [f_entity_id]) not in self.allowed_paths:
                continue

            new_max_depth = None
            if max_depth is not None:
                new_max_depth = max_depth - 1
            self._run_dfs(entity=self.es[f_entity_id],
                          entity_path=list(entity_path),
                          all_features=all_features,
                          max_depth=new_max_depth)

        """
        Step 5 - Create dfeat features for forward relationships
        """
        # get forward relationship involving forward entities
        forward = [r for r in self.es.get_forward_relationships(entity.id)
                   if r.parent_entity.id in forward_entities and
                   r.parent_entity.id not in self.ignore_entities]

        for r in forward:
            if self.allowed_paths and tuple(entity_path + [r.parent_entity.id]) not in self.allowed_paths:
                continue
            self._build_forward_features(
                all_features=all_features,
                parent_entity=r.parent_entity,
                child_entity=r.child_entity,
                relationship=r,
                max_depth=max_depth)

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
        # check if new_feature is already added: "in" uses the __eq__ function
        # so this will work.

        if (self.max_hlevel is not None and
                self._max_hlevel(new_feature) > self.max_hlevel):
            return
        entity_id = new_feature.entity.id
        if new_feature.hash() in all_features[entity_id]:
            return
            raise Exception("DFS runtime error: tried to add feature %s"
                            " more than once" % (new_feature.get_name()))

        all_features[entity_id][new_feature.hash()] = new_feature

    def _add_identity_features(self, all_features, entity):
        """converts all variables from the given entity into features

        Args:
            all_features (dict[Entity.id -> dict[str -> BaseFeature]]):
                Dict containing a dict for each entity. Each nested dict
                has features as values with their ids as keys.
            entity (Entity): Entity to calculate features for.
        """
        variables = entity.variables
        ignore_variables = self.ignore_variables[entity.id]
        for v in variables:
            if v.id in ignore_variables:
                continue
            new_f = IdentityFeature(variable=v)
            self._handle_new_feature(all_features=all_features,
                                     new_feature=new_f)

        # add seed features, if any, for dfs to build on top of
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
            if variable.interesting_values is None:
                continue

            for val in variable.interesting_values:
                self.where_clauses[entity.id].add(feat == val)

    def _build_transform_features(self, all_features, entity, max_depth=0):
        """Creates trans_features for all the variables in an entity

        Args:
            all_features (dict[:class:`.Entity`.id:dict->[str->:class:`BaseFeature`]]):
                Dict containing a dict for each entity. Each nested dict
                has features as values with their ids as keys

          entity (Entity): Entity to calculate features for.
        """
        if max_depth is not None and max_depth < 0:
            return

        new_max_depth = None
        if max_depth is not None:
            new_max_depth = max_depth - 1

        self._add_identity_features(all_features, entity)

        for trans_prim in self.trans_primitives:
            # if multiple input_types, only use first one for DFS
            input_types = trans_prim.input_types
            if type(input_types[0]) == list:
                input_types = input_types[0]

            features = self._features_by_type(all_features=all_features,
                                              entity=entity,
                                              max_depth=new_max_depth,
                                              variable_type=set(input_types))

            matching_inputs = match(input_types, features,
                                    commutative=trans_prim.commutative)

            for matching_input in matching_inputs:
                if all(bf.number_output_features == 1 for bf in matching_input):
                    new_f = TransformFeature(matching_input,
                                             primitive=trans_prim)
                    self._handle_new_feature(all_features=all_features,
                                             new_feature=new_f)

        for groupby_prim in self.groupby_trans_primitives:
            # Normally input_types is a list of what inputs can be supplied to
            # the primitive function.  Here we temporarily add `Id` as an extra
            # item in input_types so that the matching function will also look
            # for feature columns to group by.
            input_types = groupby_prim.input_types[:]
            # if multiple input_types, only use first one for DFS
            if type(input_types[0]) == list:
                input_types = input_types[0]
            input_types.append(Id)

            features = self._features_by_type(all_features=all_features,
                                              entity=entity,
                                              max_depth=new_max_depth,
                                              variable_type=set(input_types))
            matching_inputs = match(input_types, features,
                                    commutative=groupby_prim.commutative)
            for matching_input in matching_inputs:
                if all(bf.number_output_features == 1 for bf in matching_input):
                    new_f = GroupByTransformFeature(list(matching_input[:-1]),
                                                    groupby=matching_input[-1],
                                                    primitive=groupby_prim)
                    self._handle_new_feature(all_features=all_features,
                                             new_feature=new_f)

    def _build_forward_features(self, all_features, parent_entity,
                                child_entity, relationship, max_depth=0):

        if max_depth is not None and max_depth < 0:
            return

        features = self._features_by_type(
            all_features=all_features,
            entity=parent_entity,
            max_depth=max_depth,
            variable_type=variable_types.PandasTypes._all)

        for f in features:
            if self._feature_in_relationship_path([relationship], f):
                continue

            # limits allowing direct features of agg_feats with where clauses
            if isinstance(f, AggregationFeature):
                deep_base_features = [f] + f.get_dependencies(deep=True)
                for feat in deep_base_features:
                    if isinstance(feat, AggregationFeature) and feat.where is not None:
                        continue

            new_f = DirectFeature(f, child_entity)

            self._handle_new_feature(all_features=all_features,
                                     new_feature=new_f)

    def _build_agg_features(self, all_features,
                            parent_entity, child_entity, max_depth=0):
        if max_depth is not None and max_depth < 0:
            return

        new_max_depth = None
        if max_depth is not None:
            new_max_depth = max_depth - 1

        for agg_prim in self.agg_primitives:
            # if multiple input_types, only use first one for DFS
            input_types = agg_prim.input_types
            if type(input_types[0]) == list:
                input_types = input_types[0]

            features = self._features_by_type(all_features=all_features,
                                              entity=child_entity,
                                              max_depth=new_max_depth,
                                              variable_type=set(input_types))

            # remove features in relationship path
            relationship_path = self.es.find_backward_path(parent_entity.id,
                                                           child_entity.id)

            features = [f for f in features if not self._feature_in_relationship_path(
                relationship_path, f)]
            matching_inputs = match(input_types, features,
                                    commutative=agg_prim.commutative)
            wheres = list(self.where_clauses[child_entity.id])

            for matching_input in matching_inputs:
                if not check_stacking(agg_prim, matching_input):
                    continue
                new_f = AggregationFeature(matching_input,
                                           parent_entity=parent_entity,
                                           primitive=agg_prim)
                self._handle_new_feature(new_f, all_features)

                # Obey allow where
                if not agg_prim.allow_where:
                    continue

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
                    base_hashes = [f.hash() for f in new_f.base_features]
                    if any([base_feat.hash() in base_hashes for base_feat in where.base_features]):
                        continue

                    new_f = AggregationFeature(matching_input,
                                               parent_entity=parent_entity,
                                               where=where,
                                               primitive=agg_prim)

                    self._handle_new_feature(new_f, all_features)

    def _features_by_type(self, all_features, entity, max_depth,
                          variable_type=None):

        selected_features = []

        if max_depth is not None and max_depth < 0:
            return selected_features

        for feat in all_features[entity.id]:
            f = all_features[entity.id][feat]

            if (variable_type == variable_types.PandasTypes._all or
                    f.variable_type == variable_type or
                    any(issubclass(f.variable_type, vt) for vt in variable_type)):
                if ((max_depth is None or self._get_depth(f) <= max_depth) and
                        (self.max_hlevel is None or
                         self._max_hlevel(f) <= self.max_hlevel)):
                    selected_features.append(f)

        return selected_features

    def _get_depth(self, f):
        return f.get_depth(stop_at=self.seed_features)

    def _feature_in_relationship_path(self, relationship_path, feature):
        # must be identity feature to be in the relationship path
        if not isinstance(feature, IdentityFeature):
            return False

        for relationship in relationship_path:
            if relationship.child_entity.id == feature.entity.id and \
               relationship.child_variable.id == feature.variable.id:
                return True

            if relationship.parent_entity.id == feature.entity.id and \
               relationship.parent_variable.id == feature.variable.id:
                return True

        return False

    def _max_hlevel(self, f):
        # for each base_feat along each path in f,
        # if base_feat is a direct_feature of an agg_primitive
        # determine aggfeat's hlevel
        # return max hlevel
        deps = [f] + f.get_dependencies(deep=True)
        hlevel = 0
        for d in deps:
            if isinstance(d, DirectFeature) and \
                    isinstance(d.base_features[0], AggregationFeature):

                assert d.parent_entity.id == d.base_features[0].entity.id
                path, new_hlevel = self.es.find_path(self.target_entity_id,
                                                     d.parent_entity.id,
                                                     include_num_forward=True)
                hlevel = max(hlevel, new_hlevel)
        return hlevel


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


def match(input_types, features, replace=False, commutative=False):
    to_match = input_types[0]
    matches = match_by_type(features, to_match)

    if len(input_types) == 1:
        return [(m,) for m in matches]

    matching_inputs = set([])

    for m in matches:
        copy = features[:]

        if not replace:
            copy = [c for c in copy if c.hash() != m.hash()]

        rest = match(input_types[1:], copy, replace)
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
        return set([tuple(sorted(s, key=lambda x: x.get_name().lower())) for s in matching_inputs])

    return set([tuple(s) for s in matching_inputs])


def handle_primitive(primitive):
    if not isinstance(primitive, PrimitiveBase):
        primitive = primitive()
    assert isinstance(primitive, PrimitiveBase), "must be a primitive"
    return primitive


def check_trans_primitive(primitive):
    trans_prim_dict = primitives.get_transform_primitives()

    if is_string(primitive):
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
