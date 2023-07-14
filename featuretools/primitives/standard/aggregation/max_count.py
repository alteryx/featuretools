import numpy as np
from woodwork.column_schema import ColumnSchema

from featuretools.primitives.base import AggregationPrimitive


class MaxCount(AggregationPrimitive):
    """Calculates the number of occurrences of the max value in a list

    Args:
        skipna (bool): Determines if to use NA/null values. Defaults to
            True to skip NA/null. If skipna is False, and there are NaN
            values in the array, the max will be NaN regardless of
            the other values, and NaN will be returned.

    Examples:
        >>> max_count = MaxCount()
        >>> max_count([1, 2, 5, 1, 5, 3, 5])
        3

        You can optionally specify how to handle NaN values

        >>> max_count_skipna = MaxCount(skipna=False)
        >>> max_count_skipna([1, 2, 5, 1, 5, 3, None])
        nan
    """

    name = "max_count"
    input_types = [ColumnSchema(semantic_tags={"numeric"})]
    return_type = ColumnSchema(semantic_tags={"numeric"})

    def __init__(self, skipna=True):
        self.skipna = skipna

    def get_function(self):
        def max_count(x):
            xmax = x.max(skipna=self.skipna)
            if np.isnan(xmax):
                return np.nan
            return x.eq(xmax).sum()

        return max_count
