from .primitive_base import PrimitiveBase


class GroupByTransformPrimitive(PrimitiveBase):
    """TODO: docstring explaining GroupByTransformPrimitives"""
    uses_full_entity = False
    groups = []

    def generate_name(self, base_feature_names):
        name = u"{}({} by {})".format(self.name.upper(),
                                      base_feature_names[0],
                                      self.groups)
        return name
