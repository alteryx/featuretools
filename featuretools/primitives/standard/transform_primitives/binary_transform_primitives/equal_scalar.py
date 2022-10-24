from woodwork.column_schema import ColumnSchema
from woodwork.logical_types import BooleanNullable

from featuretools.primitives.base.transform_primitive_base import TransformPrimitive
from featuretools.utils.gen_utils import Library


class EqualScalar(TransformPrimitive):
    """Determines if values in a list are equal to a given scalar.

    Description:
        Given a list of values and a constant scalar, determine
        whether each of the values is equal to the scalar.

    Examples:
        >>> equal_scalar = EqualScalar(value=2)
        >>> equal_scalar([3, 1, 2]).tolist()
        [False, False, True]
    """

    name = "equal_scalar"
    input_types = [ColumnSchema()]
    return_type = ColumnSchema(logical_type=BooleanNullable)
    compatibility = [Library.PANDAS, Library.DASK, Library.SPARK]

    def __init__(self, value=None):
        self.value = value
        self.description_template = "whether {{}} equals {}".format(self.value)

    def get_function(self):
        def equal_scalar(vals):
            return vals == self.value

        return equal_scalar

    def generate_name(self, base_feature_names):
        return "%s = %s" % (base_feature_names[0], str(self.value))
