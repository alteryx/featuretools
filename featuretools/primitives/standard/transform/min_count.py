import numpy as np
from woodwork.column_schema import ColumnSchema
from woodwork.logical_types import IntegerNullable

from featuretools.primitives.base import AggregationPrimitive


class MinCount(AggregationPrimitive):
    """Calculates the number of occurrences of the min value in a list

    Args:
        skipna (bool): Determines if to use NA/null values. Defaults to
            True to skip NA/null. If skipna is False, and there are NaN
            values in the array, the min will be NaN regardless of
            the other values, and NaN will be returned.

    Examples:
        >>> min_count = MinCount()
        >>> min_count([1, 2, 5, 1, 5, 3, 5])
        2

        You can optionally specify how to handle NaN values

        >>> min_count_skipna = MinCount(skipna=False)
        >>> min_count_skipna([1, 2, 5, 1, 5, 3, None])
        nan
    """

    name = "min_count"
    input_types = [ColumnSchema(semantic_tags={"numeric"})]
    return_type = ColumnSchema(logical_type=IntegerNullable, semantic_tags={"numeric"})

    def __init__(self, skipna=True):
        self.skipna = skipna

    def get_function(self):
        def min_count(x):
            xmin = x.min(skipna=self.skipna)
            if np.isnan(xmin):
                return np.nan
            return x.eq(xmin).sum()

        return min_count
