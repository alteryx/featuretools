from woodwork.column_schema import ColumnSchema
from woodwork.logical_types import BooleanNullable

from featuretools.primitives.base.transform_primitive_base import TransformPrimitive
from featuretools.primitives.standard.transform.binary._categorical_utils import (
    align_categoricals,
)


class Equal(TransformPrimitive):
    """Determines if values in one list are equal to another list.

    Description:
        Given a list of values X and a list of values Y, determine
        whether each value in X is equal to each corresponding value
        in Y.

    Examples:
        >>> equal = Equal()
        >>> equal([2, 1, 2], [1, 2, 2]).tolist()
        [False, False, True]
    """

    name = "equal"
    input_types = [ColumnSchema(), ColumnSchema()]
    return_type = ColumnSchema(logical_type=BooleanNullable)
    commutative = True

    description_template = "whether {} equals {}"

    def get_function(self):
        def equal(x_vals, y_vals):
            x_vals, y_vals = align_categoricals(x_vals, y_vals)
            return x_vals.eq(y_vals)

        return equal

    def generate_name(self, base_feature_names):
        return "%s = %s" % (base_feature_names[0], base_feature_names[1])
