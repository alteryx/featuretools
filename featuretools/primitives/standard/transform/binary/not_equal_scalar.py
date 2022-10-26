from woodwork.column_schema import ColumnSchema
from woodwork.logical_types import BooleanNullable

from featuretools.primitives.base.transform_primitive_base import TransformPrimitive
from featuretools.utils.gen_utils import Library


class NotEqualScalar(TransformPrimitive):
    """Determines if values in a list are not equal to a given scalar.

    Description:
        Given a list of values and a constant scalar, determine
        whether each of the values is not equal to the scalar.

    Examples:
        >>> not_equal_scalar = NotEqualScalar(value=2)
        >>> not_equal_scalar([3, 1, 2]).tolist()
        [True, True, False]
    """

    name = "not_equal_scalar"
    input_types = [ColumnSchema()]
    return_type = ColumnSchema(logical_type=BooleanNullable)
    compatibility = [Library.PANDAS, Library.DASK, Library.SPARK]

    def __init__(self, value=None):
        self.value = value
        self.description_template = "whether {{}} does not equal {}".format(self.value)

    def get_function(self):
        def not_equal_scalar(vals):
            return vals != self.value

        return not_equal_scalar

    def generate_name(self, base_feature_names):
        return "%s != %s" % (base_feature_names[0], str(self.value))
