import numpy as np
from woodwork.column_schema import ColumnSchema
from woodwork.logical_types import Boolean, BooleanNullable

from featuretools.primitives.base.transform_primitive_base import TransformPrimitive
from featuretools.utils.gen_utils import Library


class MultiplyBoolean(TransformPrimitive):
    """Performs element-wise multiplication of two lists of boolean values.

    Description:
        Given a list of boolean values X and a list of boolean
        values Y, determine the product of each value in X
        with its corresponding value in Y.

    Examples:
        >>> multiply_boolean = MultiplyBoolean()
        >>> multiply_boolean([True, True, False], [True, False, True]).tolist()
        [True, False, False]
    """

    name = "multiply_boolean"
    input_types = [
        [
            ColumnSchema(logical_type=BooleanNullable),
            ColumnSchema(logical_type=BooleanNullable),
        ],
        [ColumnSchema(logical_type=Boolean), ColumnSchema(logical_type=Boolean)],
        [
            ColumnSchema(logical_type=Boolean),
            ColumnSchema(logical_type=BooleanNullable),
        ],
        [
            ColumnSchema(logical_type=BooleanNullable),
            ColumnSchema(logical_type=Boolean),
        ],
    ]
    return_type = ColumnSchema(logical_type=BooleanNullable)
    commutative = True
    compatibility = [Library.PANDAS, Library.DASK]
    description_template = "the product of {} and {}"

    def get_function(self):
        return np.bitwise_and

    def generate_name(self, base_feature_names):
        return "%s * %s" % (base_feature_names[0], base_feature_names[1])
