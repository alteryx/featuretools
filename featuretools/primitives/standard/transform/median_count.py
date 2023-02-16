import numpy as np
from woodwork.column_schema import ColumnSchema
from woodwork.logical_types import IntegerNullable

from featuretools.primitives.base import AggregationPrimitive


class MedianCount(AggregationPrimitive):
    """Calculates the number of occurrences of the median value in a list

    Args:
        skipna (bool): Determines if to use NA/null values. Defaults to
            True to skip NA/null. If skipna is False, and there are NaN
            values in the array, the median will be NaN, regardless of
            the other values.

    Examples:
        >>> median_count = MedianCount()
        >>> median_count([1, 2, 3, 1, 5, 3, 5])
        2

        You can optionally specify how to handle NaN values

        >>> median_count_skipna = MedianCount(skipna=False)
        >>> median_count_skipna([1, 2, 3, 1, 5, 3, None])
        nan
    """

    name = "median_count"
    input_types = [ColumnSchema(semantic_tags={"numeric"})]
    return_type = ColumnSchema(logical_type=IntegerNullable, semantic_tags={"numeric"})
    stack_on_self = False
    default_value = 0

    def __init__(self, skipna=True):
        self.skipna = skipna

    def get_function(self):
        def median_count(x):
            median = x.median(skipna=self.skipna)
            if np.isnan(median):
                return np.nan
            return x.eq(median).sum()

        return median_count
