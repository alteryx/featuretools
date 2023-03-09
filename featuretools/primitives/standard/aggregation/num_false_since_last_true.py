import numpy as np
from woodwork.column_schema import ColumnSchema
from woodwork.logical_types import Boolean, IntegerNullable

from featuretools.primitives.base import AggregationPrimitive


class NumFalseSinceLastTrue(AggregationPrimitive):
    """Calculates the number of 'False' values since the last `True` value.
    Description:
        From a series of Booleans, find the last record with a `True` value.
        Return the count of 'False' values between that record and the end of
        the series. Return nan if no values are `True`. Any nan values in the
        input are ignored. A 'True' value in the last row will result in a
        count of 0.  Inputs are converted too booleans before calculating
        the result.
    Examples:
        >>> num_false_since_last_true = NumFalseSinceLastTrue()
        >>> num_false_since_last_true([True, False, True, False, False])
        2
    """

    name = "num_false_since_last_true"
    input_types = [ColumnSchema(logical_type=Boolean)]
    return_type = ColumnSchema(logical_type=IntegerNullable, semantic_tags={"numeric"})
    stack_on_self = False
    default_value = 0

    def get_function(self):
        def num_false_since_last_true(x):
            if x.empty:
                return np.nan
            x = x.dropna().astype(bool)
            true_indices = x[x]
            if true_indices.empty:
                return np.nan
            last_true_index = true_indices.index[-1]
            x_slice = x.loc[last_true_index:]
            return np.invert(x_slice).sum()

        return num_false_since_last_true
