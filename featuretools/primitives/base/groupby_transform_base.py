from .primitive_base import PrimitiveBase


class GroupByTransformPrimitive(PrimitiveBase):
    """TODO: docstring explaining GroupByTransformPrimitives"""
    uses_full_entity = False

    def generate_name(self, base_feature_names, groupby):
        name = u"{}({} by {})".format(self.name.upper(),
                                      base_feature_names[0],
                                      groupby)
        return name
