import itertools
import logging
from builtins import object
from collections import defaultdict

from featuretools.feature_base import (
    AggregationFeature,
    GroupByTransformFeature,
    TransformFeature
)
from featuretools.utils import Trie

logger = logging.getLogger('featuretools.computational_backend')


class FeatureSet(object):
    def __init__(self, entityset, features, ignored=None):
        self.entityset = entityset
        self.target_eid = features[0].entity.id
        self.target_features = features
        self.target_feature_names = {f.unique_name() for f in features}
        if ignored is None:
            self.ignored = set([])
        else:
            self.ignored = ignored

        # Maps the unique name of each feature to the actual feature. This is necessary
        # because features do not support equality and so cannot be used as
        # dictionary keys. The equality operator on features produces a new
        # feature (which will always be truthy).
        self.features_by_name = {f.unique_name(): f for f in features}

        feature_dependents = defaultdict(set)
        for f in features:
            deps = f.get_dependencies(deep=True, ignored=ignored)
            for dep in deps:
                feature_dependents[dep.unique_name()].add(f.unique_name())
                self.features_by_name[dep.unique_name()] = dep
                subdeps = dep.get_dependencies(deep=True, ignored=ignored)
                for sd in subdeps:
                    feature_dependents[sd.unique_name()].add(dep.unique_name())

        # feature names (keys) and the features that rely on them (values).
        self.feature_dependents = {
            fname: [self.features_by_name[dname] for dname in feature_dependents[fname]]
            for fname, f in self.features_by_name.items()}

        self.feature_trie = self._build_feature_trie()

    def _build_feature_trie(self):
        """
        Construct a Trie mapping relationship paths to sets of features.

        The edges of the tree are tuples of (is_forward (bool), relationship).
        The values are sets of feature.unique_name
        """
        feature_trie = Trie(default=set)

        for f in self.target_features:
            self._add_feature_to_trie(feature_trie, f, [])

        return feature_trie

    def _add_feature_to_trie(self, trie, f, path):
        sub_trie = trie.get_node(path)
        sub_trie[[]].add(f.unique_name())
        sub_path = f.relationship_path
        if sub_path:
            sub_path = [(f.path_is_forward, r) for r in sub_path]

        for dep_feat in f.get_dependencies():
            self._add_feature_to_trie(sub_trie, dep_feat, sub_path)

    def group_features(self, feature_names):
        """
        Topologically sort the given features, then group by path,
        feature type, use_previous, and where.
        """
        features = [self.features_by_name[name] for name in feature_names]
        depths = self._get_feature_depths(features)

        def key_func(f):
            return (depths[f.unique_name()],
                    f.relationship_path_name(),
                    str(f.__class__),
                    _get_use_previous(f),
                    _get_where(f),
                    f.primitive.uses_full_entity,
                    self._output_frames_type(f),
                    _get_groupby(f))

        # Sort the list of features by the complex key function above, then
        # group them by the same key
        sort_feats = sorted(features, key=key_func)
        feature_groups = [list(g) for _, g in
                          itertools.groupby(sort_feats, key=key_func)]

        return feature_groups

    def _get_feature_depths(self, features):
        """
        Generate and return a mapping of {feature name -> depth} in the
        feature DAG for the given entity.
        """
        order = defaultdict(int)
        depths = {}
        queue = features[:]
        while queue:
            # Get the next feature.
            f = queue.pop(0)

            depths[f.unique_name()] = order[f.unique_name()]

            # Only look at dependencies if they are on the same entity.
            if not f.relationship_path:
                dependencies = f.get_dependencies(ignored=self.ignored)
                for dep in dependencies:
                    order[dep.unique_name()] = \
                        min(order[f.unique_name()] - 1, order[dep.unique_name()])
                    queue.append(dep)

        return depths

    def uses_full_entity(self, feature):
        if (isinstance(feature, (GroupByTransformFeature, TransformFeature)) and
                feature.primitive.uses_full_entity):
            return True
        return self._dependent_uses_full_entity(feature)

# These functions are used for sorting and grouping features

    def _output_frames_type(self, f):
        is_output = f.unique_name() in self.target_feature_names
        dependent_uses_full_entity = self._dependent_uses_full_entity(f)
        dependent_has_subset_input = any([(not isinstance(d, TransformFeature) or
                                           (not d.primitive.uses_full_entity and
                                            d.unique_name() in self.target_feature_names))
                                          for d in self.feature_dependents[f.unique_name()]])
        # If the feature is one in which the user requested as
        # an output (meaning it's part of the input feature list
        # to calculate_feature_matrix), or a dependent feature
        # takes the subset entity_frames as input, then we need
        # to subselect the output based on the desired instance ids
        # and place in the return data frame.
        if dependent_uses_full_entity and is_output:
            return 'full_and_subset_entity_frames'
        elif dependent_uses_full_entity and dependent_has_subset_input:
            return 'full_and_subset_entity_frames'
        elif dependent_uses_full_entity:
            return 'full_entity_frames'
        # If the feature itself does not require all the instance values
        # or no dependent features do, then we
        # subselect the output
        # to only desired instances
        return 'subset_entity_frames'

    def _dependent_uses_full_entity(self, feature):
        for d in self.feature_dependents[feature.unique_name()]:
            if isinstance(d, TransformFeature) and d.primitive.uses_full_entity:
                return True
        return False


def _get_use_previous(f):
    if isinstance(f, AggregationFeature) and f.use_previous is not None:
        return (f.use_previous.unit, f.use_previous.value)
    else:
        return ("", -1)


def _get_where(f):
    if isinstance(f, AggregationFeature) and f.where is not None:
        return f.where.unique_name()
    else:
        return ''


def _get_groupby(f):
    if isinstance(f, GroupByTransformFeature):
        return f.groupby.unique_name()
    else:
        return ''
