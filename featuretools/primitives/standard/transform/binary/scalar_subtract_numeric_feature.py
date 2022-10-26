from woodwork.column_schema import ColumnSchema

from featuretools.primitives.base.transform_primitive_base import TransformPrimitive
from featuretools.utils.gen_utils import Library


class ScalarSubtractNumericFeature(TransformPrimitive):
    """Subtracts each value in the list from a given scalar.

    Description:
        Given a list of numeric values and a scalar, subtract
        the each value from the scalar and return the list of
        differences.

    Examples:
        >>> scalar_subtract_numeric_feature = ScalarSubtractNumericFeature(value=2)
        >>> scalar_subtract_numeric_feature([3, 1, 2]).tolist()
        [-1, 1, 0]
    """

    name = "scalar_subtract_numeric_feature"
    input_types = [ColumnSchema(semantic_tags={"numeric"})]
    return_type = ColumnSchema(semantic_tags={"numeric"})
    compatibility = [Library.PANDAS, Library.DASK, Library.SPARK]

    def __init__(self, value=0):
        self.value = value
        self.description_template = "the result {} minus {{}}".format(self.value)

    def get_function(self):
        def scalar_subtract_numeric_feature(vals):
            return self.value - vals

        return scalar_subtract_numeric_feature

    def generate_name(self, base_feature_names):
        return "%s - %s" % (str(self.value), base_feature_names[0])
