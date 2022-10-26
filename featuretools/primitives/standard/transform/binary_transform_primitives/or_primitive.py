import numpy as np
from woodwork.column_schema import ColumnSchema
from woodwork.logical_types import Boolean, BooleanNullable

from featuretools.primitives.base.transform_primitive_base import TransformPrimitive
from featuretools.utils.gen_utils import Library


class Or(TransformPrimitive):
    """Performs element-wise logical OR of two lists.

    Description:
        Given a list of booleans X and a list of booleans Y,
        determine whether each value in X is `True`, or
        whether its corresponding value in Y is `True`.

    Examples:
        >>> _or = Or()
        >>> _or([False, True, False], [True, True, False]).tolist()
        [True, True, False]
    """

    name = "or"
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
    compatibility = [Library.PANDAS, Library.DASK, Library.SPARK]
    description_template = "whether {} is true or {} is true"

    def get_function(self):
        return np.logical_or

    def generate_name(self, base_feature_names):
        return "OR(%s, %s)" % (base_feature_names[0], base_feature_names[1])
