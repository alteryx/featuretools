from woodwork.column_schema import ColumnSchema

from featuretools.primitives.base.transform_primitive_base import TransformPrimitive
from featuretools.utils.gen_utils import Library


class ModuloByFeature(TransformPrimitive):
    """Computes the modulo of a scalar by each element in a list.

    Description:
        Given a list of numeric values and a scalar, return the
        modulo, or remainder of the scalar after being divided
        by each value.

    Examples:
        >>> modulo_by_feature = ModuloByFeature(value=2)
        >>> modulo_by_feature([4, 1, 2]).tolist()
        [2, 0, 0]
    """

    name = "modulo_by_feature"
    input_types = [ColumnSchema(semantic_tags={"numeric"})]
    return_type = ColumnSchema(semantic_tags={"numeric"})
    compatibility = [Library.PANDAS, Library.DASK, Library.SPARK]

    def __init__(self, value=1):
        self.value = value
        self.description_template = "the remainder after dividing {} by {{}}".format(
            self.value,
        )

    def get_function(self):
        def modulo_by_feature(vals):
            return self.value % vals

        return modulo_by_feature

    def generate_name(self, base_feature_names):
        return "%s %% %s" % (str(self.value), base_feature_names[0])
