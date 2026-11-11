from woodwork.column_schema import ColumnSchema
from woodwork.logical_types import BooleanNullable

from featuretools.primitives.base.transform_primitive_base import TransformPrimitive
from featuretools.primitives.standard.transform.binary._categorical_utils import (
    align_categoricals,
)


class NotEqual(TransformPrimitive):
    """Determines if values in one list are not equal to another list.

    Description:
        Given a list of values X and a list of values Y, determine
        whether each value in X is not equal to each corresponding
        value in Y.

    Examples:
        >>> not_equal = NotEqual()
        >>> not_equal([2, 1, 2], [1, 2, 2]).tolist()
        [True, True, False]
    """

    name = "not_equal"
    input_types = [ColumnSchema(), ColumnSchema()]
    return_type = ColumnSchema(logical_type=BooleanNullable)
    commutative = True
    description_template = "whether {} does not equal {}"

    def get_function(self):
        def not_equal(x_vals, y_vals):
            x_vals, y_vals = align_categoricals(x_vals, y_vals)
            return x_vals.ne(y_vals)

        return not_equal

    def generate_name(self, base_feature_names):
        return "%s != %s" % (base_feature_names[0], base_feature_names[1])
