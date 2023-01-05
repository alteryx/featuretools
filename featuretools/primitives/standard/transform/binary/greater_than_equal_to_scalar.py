from woodwork.column_schema import ColumnSchema
from woodwork.logical_types import BooleanNullable

from featuretools.primitives.base.transform_primitive_base import TransformPrimitive
from featuretools.utils.gen_utils import Library


class GreaterThanEqualToScalar(TransformPrimitive):
    """Determines if values are greater than or equal to a given scalar.

    Description:
        Given a list of values and a constant scalar, determine
        whether each of the values is greater than or equal to the
        scalar. If a value is equal to the scalar, return `True`.

    Examples:
        >>> greater_than_equal_to_scalar = GreaterThanEqualToScalar(value=2)
        >>> greater_than_equal_to_scalar([3, 1, 2]).tolist()
        [True, False, True]
    """

    name = "greater_than_equal_to_scalar"
    input_types = [ColumnSchema(semantic_tags={"numeric"})]
    return_type = ColumnSchema(logical_type=BooleanNullable)
    compatibility = [Library.PANDAS, Library.DASK, Library.SPARK]

    def __init__(self, value=0):
        self.value = value
        self.description_template = (
            "whether {{}} is greater than or equal to {}".format(self.value)
        )

    def get_function(self):
        def greater_than_equal_to_scalar(vals):
            return vals >= self.value

        return greater_than_equal_to_scalar

    def generate_name(self, base_feature_names):
        return "%s >= %s" % (base_feature_names[0], str(self.value))
