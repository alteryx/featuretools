import numpy as np
import pandas as pd
import pandas.api.types as pdtypes
from woodwork.column_schema import ColumnSchema
from woodwork.logical_types import Boolean, BooleanNullable, Datetime, Ordinal

from featuretools.primitives.core.transform_primitive import TransformPrimitive
from featuretools.utils.gen_utils import Library

class And(TransformPrimitive):
    """Element-wise logical AND of two lists.

    Description:
        Given a list of booleans X and a list of booleans Y,
        determine whether each value in X is `True`, and
        whether its corresponding value in Y is also `True`.

    Examples:
        >>> _and = And()
        >>> _and([False, True, False], [True, True, False]).tolist()
        [False, True, False]
    """

    name = "and"
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
    description_template = "whether {} and {} are true"

    def get_function(self):
        return np.logical_and

    def generate_name(self, base_feature_names):
        return "AND(%s, %s)" % (base_feature_names[0], base_feature_names[1])