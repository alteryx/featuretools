from .primitive_base import PrimitiveBase

from featuretools.variable_types import Variable


class DirectFeature(PrimitiveBase):
    """Feature for child entity that inherits
        a feature value from a parent entity"""
    input_types = [Variable]
    return_type = None

    def __init__(self, base_feature, child_entity):
        base_feature = self._check_feature(base_feature)
        if base_feature.expanding:
            self.expanding = True

        path = child_entity.entityset.find_forward_path(child_entity.id, base_feature.entity.id)
        if len(path) > 1:
            parent_entity_id = path[1].child_entity.id
            parent_entity = child_entity.entityset[parent_entity_id]
            parent_feature = DirectFeature(base_feature, parent_entity)
        else:
            parent_feature = base_feature

        self.parent_entity = parent_feature.entity
        self._variable_type = parent_feature.variable_type
        super(DirectFeature, self).__init__(child_entity, [parent_feature])

    @property
    def default_value(self):
        return self.base_features[0].default_value

    def generate_name(self):
        return u"%s.%s" % (self.parent_entity.id,
                           self.base_features[0].get_name())
