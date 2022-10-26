import numpy as np
from woodwork.column_schema import ColumnSchema
from woodwork.logical_types import Boolean, BooleanNullable

from featuretools.primitives.base import TransformPrimitive
from featuretools.utils.gen_utils import Library


class Not(TransformPrimitive):
    """Negates a boolean value.

    Examples:
        >>> not_func = Not()
        >>> not_func([True, True, False]).tolist()
        [False, False, True]
    """

    name = "not"
    input_types = [
        [ColumnSchema(logical_type=Boolean)],
        [ColumnSchema(logical_type=BooleanNullable)],
    ]
    return_type = ColumnSchema(logical_type=BooleanNullable)
    compatibility = [Library.PANDAS, Library.DASK, Library.SPARK]
    description_template = "the negation of {}"

    def generate_name(self, base_feature_names):
        return "NOT({})".format(base_feature_names[0])

    def get_function(self):
        return np.logical_not
