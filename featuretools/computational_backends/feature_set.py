import itertools
import logging
from builtins import object
from collections import defaultdict

from featuretools.entityset.relationship import RelationshipPath
from featuretools.feature_base import (
    AggregationFeature,
    GroupByTransformFeature,
    TransformFeature
)
from featuretools.utils import Trie

logger = logging.getLogger('featuretools.computational_backend')


class FeatureSet(object):
    def __init__(self, features):
        self.target_eid = features[0].entity.id
        self.target_features = features
        self.target_feature_names = {f.unique_name() for f in features}

        # Maps the unique name of each feature to the actual feature. This is necessary
        # because features do not support equality and so cannot be used as
        # dictionary keys. The equality operator on features produces a new
        # feature (which will always be truthy).
        self.features_by_name = {f.unique_name(): f for f in features}

        feature_dependents = defaultdict(set)
        for f in features:
            deps = f.get_dependencies(deep=True)
            for dep in deps:
                feature_dependents[dep.unique_name()].add(f.unique_name())
                self.features_by_name[dep.unique_name()] = dep
                subdeps = dep.get_dependencies(deep=True)
                for sd in subdeps:
                    feature_dependents[sd.unique_name()].add(dep.unique_name())

        # feature names (keys) and the features that rely on them (values).
        self.feature_dependents = {
            fname: [self.features_by_name[dname] for dname in feature_dependents[fname]]
            for fname, f in self.features_by_name.items()}

        self.feature_trie = self._build_feature_trie()

    def _build_feature_trie(self):
        """
        Construct a trie mapping RelationshipPaths to sets of feature names.
        """
        feature_trie = Trie(default=set, path_constructor=RelationshipPath)

        for f in self.target_features:
            self._add_feature_to_trie(feature_trie, f, [])

        return feature_trie

    def _add_feature_to_trie(self, trie, f, path):
        sub_trie = trie.get_node(path)
        sub_trie.value.add(f.unique_name())

        for dep_feat in f.get_dependencies():
            self._add_feature_to_trie(sub_trie, dep_feat, f.relationship_path)

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
                    self.uses_full_entity(f),
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
                dependencies = f.get_dependencies()
                for dep in dependencies:
                    order[dep.unique_name()] = \
                        min(order[f.unique_name()] - 1, order[dep.unique_name()])
                    queue.append(dep)

        return depths

    def uses_full_entity(self, feature, check_dependents=False):
        if isinstance(feature, TransformFeature) and feature.primitive.uses_full_entity:
            return True
        return check_dependents and self._dependent_uses_full_entity(feature)

    def _dependent_uses_full_entity(self, feature):
        for d in self.feature_dependents[feature.unique_name()]:
            if isinstance(d, TransformFeature) and d.primitive.uses_full_entity:
                return True
        return False

# These functions are used for sorting and grouping features


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
