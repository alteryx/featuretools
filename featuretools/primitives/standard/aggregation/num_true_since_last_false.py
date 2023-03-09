import numpy as np
from woodwork.column_schema import ColumnSchema
from woodwork.logical_types import Boolean, IntegerNullable

from featuretools.primitives.base import AggregationPrimitive


class NumTrueSinceLastFalse(AggregationPrimitive):
    """Calculates the number of 'True' values since the last `False` value.
    Description:
        From a series of Booleans, find the last record with a `False` value.
        Return the count of 'True' values between that record and the end of
        the series. Return nan if no values are `False`. Any nan values in the
        input are ignored. A 'False' value in the last row will result in a
        count of 0.
    Examples:
        >>> num_true_since_last_false = NumTrueSinceLastFalse()
        >>> num_true_since_last_false([False, True, False, True, True])
        2
    """

    name = "num_true_since_last_false"
    input_types = [ColumnSchema(logical_type=Boolean)]
    return_type = ColumnSchema(logical_type=IntegerNullable, semantic_tags={"numeric"})
    stack_on_self = False
    default_value = 0

    def get_function(self):
        def num_true_since_last_false(x):
            x = x.dropna().astype(bool)
            false_indices = x[~x]
            if false_indices.empty:
                return np.nan
            last_false_index = false_indices.index[-1]
            x_slice = x.loc[last_false_index:]
            return x_slice.sum()

        return num_true_since_last_false
