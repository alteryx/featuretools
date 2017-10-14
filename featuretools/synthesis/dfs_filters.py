from featuretools.primitives import (AggregationPrimitive, DirectFeature,
                                     IdentityFeature, Mode, TransformPrimitive)
from featuretools.variable_types import Discrete


class DFSFilterBase(object):
    filter_type = 'post_instance'
    apply_to = [AggregationPrimitive, TransformPrimitive, DirectFeature]

    def apply_filter(self, f, e):
        raise NotImplementedError(self.__class__.__name__ + '.apply_filter')

    def post_instance_validate(self, f, e, target_entity_id):
        """
        Args:
            f (`.PrimitiveBase`): feature
            e (`.BaseEntity`): entity
            target_entity_id (str): id of target entity
        return True if this filter doesn't apply to feature f
        """
        for at in self.apply_to:
            if isinstance(f, at):
                break
        else:
            return True

        return self.apply_filter(f, e, target_entity_id)

    def is_valid(self, feature=None, entity=None,
                 target_entity_id=None, child_feature=None,
                 child_entity=None, entity_path=None, forward=None,
                 where=None):

        if self.filter_type == 'post_instance':
            args = [feature, entity, target_entity_id]
            func = self.post_instance_validate

        elif self.filter_type == 'traversal':
            args = [entity, child_entity,
                    target_entity_id, entity_path, forward]
            func = self.apply_filter

        else:
            raise NotImplementedError("Unknown filter type: {}".
                                      format(self.filter_type))

        if type(feature) != list:
            return func(*args)

        else:
            return [func(*args) for f in feature]


# P TODO move into DFS
class TraverseUp(DFSFilterBase):
    filter_type = 'traversal'

    def __init__(self, percent_low=.0005, percent_high=.9995,
                 unique_high=50, unique_low=5):
        self.percent_low = percent_low
        self.percent_high = percent_high
        self.unique_high = unique_high
        self.unique_low = unique_low

    def apply_filter(self, parent_entity, child_entity,
                     target_entity_id, entity_path, forward):
        es = parent_entity.entityset
        if not forward:
            if (parent_entity.id == target_entity_id or
                    es.find_backward_path(parent_entity.id,
                                          target_entity_id) is None):
                return True
            path = es.find_backward_path(parent_entity.id, child_entity.id)
            r = path[0]

            percent_unique = r.child_variable.percent_unique
            count = r.child_variable.count
            if (percent_unique is None or
                    count is None):
                return True

            # Traverse if high absolute number of unique
            if count > self.unique_high:
                return True

            # Don't traverse if low absolute number of unique
            if count < self.unique_low:
                return False

            # Don't traverse if not unique enough or too unique
            if (percent_unique > self.percent_high or
                    percent_unique < self.percent_low):
                return False

        return True


# P TODO: get this back in
class LimitModeUniques(DFSFilterBase):
    """Heuristic to discard mode feature if child feature values
    contain too many uniques.

    An individual child feature has too many uniques if the
    ratio of uniques to count values is greater than a threshold,
    defaulted to .9.
    """
    filter_type = 'post_instance'
    apply_to = [Mode]

    def __init__(self, threshold=.9):
        self.threshold = threshold

    def apply_filter(self, f, e, target_entity_id):
        child_feature = f.base_features[0]
        if not isinstance(child_feature, IdentityFeature):
            return True

        variable = child_feature.variable
        if not isinstance(variable, Discrete):
            return True

        percent_unique = variable.percent_unique
        if percent_unique is not None and (percent_unique > self.threshold):
            return False

        return True


filter_map = {
    "TraverseUp": TraverseUp,
}


def make_filter(name, params={}):
    return filter_map[name](**params)
