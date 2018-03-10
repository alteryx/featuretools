import copy
import itertools
import logging
from builtins import object
from collections import defaultdict

from ..utils import gen_utils as utils

from featuretools import variable_types
from featuretools.exceptions import UnknownFeature
from featuretools.primitives import (
    AggregationPrimitive,
    DirectFeature,
    IdentityFeature,
    TransformPrimitive
)

logger = logging.getLogger('featuretools.computational_backend')


class FeatureTree(object):

    def __init__(self, entityset, features, ignored=None):
        self.entityset = entityset
        self.target_eid = features[0].entity.id
        if ignored is None:
            self.ignored = set([])
        else:
            self.ignored = ignored

        self.feature_hashes = set([f.hash() for f in features])

        all_features = {f.hash(): f for f in features}
        feature_dependencies = {}
        feature_dependents = defaultdict(set)
        for f in features:
            deps = f.get_deep_dependencies(ignored=ignored)
            feature_dependencies[f.hash()] = deps
            for dep in deps:
                feature_dependents[dep.hash()].add(f.hash())
                all_features[dep.hash()] = dep
                subdeps = dep.get_deep_dependencies(
                    ignored=ignored)
                feature_dependencies[dep.hash()] = subdeps
                for sd in subdeps:
                    feature_dependents[sd.hash()].add(dep.hash())
        # turn values which were hashes of features into the features themselves
        # (note that they were hashes to prevent checking equality of feature objects,
        #  which is not currently an allowed operation)
        self.feature_dependents = {fhash: [all_features[dhash] for dhash in feature_dependents[fhash]]
                                   for fhash, f in all_features.items()}
        self.feature_dependencies = feature_dependencies
        self.all_features = list(all_features.values())
        self._find_necessary_columns()

        self._generate_feature_tree(features)
        self._order_entities()
        self._order_feature_groups()

    def _find_necessary_columns(self):
        # TODO: Can try to remove columns that are only used in the
        # intermediate large_entity_frames from self.necessary_columns
        # TODO: Can try to only keep Id/Index/DatetimeTimeIndex if actually
        # used for features
        self.necessary_columns = defaultdict(set)
        entities = set([f.entity.id for f in self.all_features])

        relationships = self.entityset.relationships
        relationship_vars = defaultdict(set)
        for r in relationships:
            relationship_vars[r.parent_entity.id].add(r.parent_variable.id)
            relationship_vars[r.child_entity.id].add(r.child_variable.id)

        for eid in entities:
            entity = self.entityset[eid]
            # Need to keep all columns used to link entities together
            # and to sort by time
            index_cols = [v.id for v in entity.variables
                          if isinstance(v, (variable_types.Index,
                                            variable_types.TimeIndex)) or
                          v.id in relationship_vars[eid]]
            self.necessary_columns[eid] |= set(index_cols)
        self.necessary_columns_for_all_values_features = copy.copy(self.necessary_columns)

        identity_features = [f for f in self.all_features
                             if isinstance(f, IdentityFeature)]

        for f in identity_features:
            self.necessary_columns[f.entity.id].add(f.variable.id)
            if self.uses_full_entity(f):
                self.necessary_columns_for_all_values_features[f.entity.id].add(f.variable.id)
        self.necessary_columns = {eid: list(cols) for eid, cols in self.necessary_columns.items()}
        self.necessary_columns_for_all_values_features = {eid: list(cols) for eid, cols in self.necessary_columns_for_all_values_features.items()}

    def _generate_feature_tree(self, features):
        """
        Given a set of features for a target entity, build a tree linking
        top-level entities in the entityset to the features that need to be
        calculated on each.
        """
        # build a set of all features, including top-level features and
        # dependencies.
        self.top_level_features = defaultdict(list)

        # find top-level features and index them by entity id.
        for f in self.all_features:
            _, num_forward = self.entityset.find_path(self.target_eid, f.entity.id,
                                                      include_num_forward=True)
            if num_forward or f.entity.id == self.target_eid:
                self.top_level_features[f.entity.id].append(f)

    def _order_entities(self):
        """
        Perform a topological sort on the top-level entities in this entityset.
        The resulting order allows each entity to be calculated after all of its
        dependencies.
        """
        entity_deps = defaultdict(set)
        for e, features in self.top_level_features.items():
            # iterate over all dependency features of the top-level features on
            # this entity. If any of these are themselves top-level features, add
            # their entities as dependencies of the current entity.
            deps = {g.hash(): g for f in features
                    for g in self.feature_dependencies[f.hash()]}
            for d in deps.values():
                _, num_forward = self.entityset.find_path(e, d.entity.id,
                                                          include_num_forward=True)
                if num_forward > 0:
                    entity_deps[e].add(d.entity.id)

        # Do a top-sort on the new entity DAG
        self.ordered_entities = utils.topsort([self.target_eid],
                                              lambda e: entity_deps[e])

    def _order_feature_groups(self):
        """
        For each top-level entity, perform a topological sort on its features,
        then group them together by entity, base entity, feature type, and the
        use_previous attribute.

        This function produces, for each toplevel entity, a list of groups of
        features which can be passed into the feature calculation functions in
        PandasBackend directly.
        """
        self.ordered_feature_groups = {}
        for entity_id in self.top_level_features:
            all_features, feature_depth = self._get_feature_depths(entity_id)

            def key_func(f):
                return (feature_depth[f.hash()],
                        f.entity.id,
                        _get_base_entity_id(f),
                        _get_ftype_string(f),
                        _get_use_previous(f),
                        _get_where(f),
                        self.uses_full_entity_differentiator(f))

            # Sort the list of features by the complex key function above, then
            # group them by the same key
            sort_feats = sorted(list(all_features), key=key_func)
            feature_groups = [list(g) for _, g in
                              itertools.groupby(sort_feats, key=key_func)]

            self.ordered_feature_groups[entity_id] = feature_groups

    def _get_feature_depths(self, entity_id):
        """
        Generate and return a mapping of {feature f -> depth of f} in the
        feature DAG for the given entity
        """
        features = {}
        order = defaultdict(int)
        out = {}
        queue = list(self.top_level_features[entity_id])
        while queue:
            # Get the next feature and make sure it's in the dict
            f = queue.pop(0)

            # stop looking if the feature we've hit is on another top-level
            # entity which is not a descendent of the current one. In this case,
            # we know we won't need to calculate this feature explicitly
            # because it should be handled by the other entity; we can treat it
            # like an identity feature.
            if f.entity.id in self.top_level_features.keys() and \
                    f.entity.id != entity_id and not \
                    self.entityset.find_backward_path(start_entity_id=entity_id,
                                                      goal_entity_id=f.entity.id):
                continue

            # otherwise, add this feature to the output dict
            out[f.hash()] = order[f.hash()]
            features[f.hash()] = f

            dependencies = f.get_dependencies(ignored=self.ignored)
            for dep in dependencies:
                order[dep.hash()] = min(order[f.hash()] - 1, order[dep.hash()])
                queue.append(dep)

        return list(features.values()), out

    def uses_full_entity(self, feature):
        if feature.uses_full_entity:
            return True
        return self._dependent_uses_full_entity(feature)

    def _dependent_uses_full_entity(self, feature):
        for d in self.feature_dependents[feature.hash()]:
            if d.uses_full_entity:
                return True
        return False

# These functions are used for sorting and grouping features

    def uses_full_entity_differentiator(self, f):
        is_output = f.hash() in self.feature_hashes
        normal_dependent_is_output = any([(not d.uses_full_entity and
                                           d.hash() in self.feature_hashes)
                                          for d in self.feature_dependents[f.hash()]])
        need_selected_output = is_output or normal_dependent_is_output
        # If a dependent feature requires all the instance values
        # of the associated entity, then we need to calculate this
        # feature on all values
        # If the feature is one in which the user requested as
        # an output (meaning it's part of the input feature list
        # to calculate_feature_matrix), or a normal, non-uses_full_entity dependent
        # feature is an output, then we also need
        # to subselect the output based on the desired instance ids
        # and place in the return data frame.
        if self._dependent_uses_full_entity(f) and need_selected_output:
            return "dependent_and_output"
        elif self._dependent_uses_full_entity(f):
            return "dependent"
        # If the feature itself requires all the instance values
        # but no dependent features do, then we need to provide
        # all the values as input, but subselect the output
        # to only desired instances
        elif self.uses_full_entity(f):
            return "uses_full_no_dependent"
        # None of the above cases, feature and accept selected
        # instances and output selected instances
        else:
            return "normal_no_dependent"


def _get_use_previous(f):
    if hasattr(f, "use_previous") and f.use_previous is not None:
        previous = f.use_previous
        return (previous.unit, previous.value)
    else:
        return ("", -1)


def _get_where(f):
    if hasattr(f, "where") and f.where is not None:
        return f.where.hash()
    else:
        return -1


def _get_base_entity_id(f):
    # return f.entity_id
    if isinstance(f, IdentityFeature):
        return f.entity_id
    else:
        # Assume all of f's base_features belong to the same entity
        return f.base_features[0].entity_id


def _get_ftype_string(f):
    if isinstance(f, TransformPrimitive):
        return "transform"
    elif isinstance(f, DirectFeature):
        return "direct"
    elif isinstance(f, AggregationPrimitive):
        return "aggregation"
    elif isinstance(f, IdentityFeature):
        return "identity"
    else:
        raise UnknownFeature("{} feature unknown".format(f.__class__))
