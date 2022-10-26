import numpy as np
from woodwork.column_schema import ColumnSchema
from woodwork.logical_types import IntegerNullable

from featuretools.primitives.base.aggregation_primitive_base import AggregationPrimitive


class CountBelowMean(AggregationPrimitive):
    """Determines the number of values that are below the mean.

    Args:
        skipna (bool): Determines if to use NA/null values. Defaults to
            True to skip NA/null.

    Examples:
        >>> count_below_mean = CountBelowMean()
        >>> count_below_mean([1, 2, 3, 4, 10])
        3

        The way NaNs are treated can be controlled.

        >>> count_below_mean_skipna = CountBelowMean(skipna=False)
        >>> count_below_mean_skipna([1, 2, 3, 4, 5, None])
        nan
    """

    name = "count_below_mean"
    input_types = [ColumnSchema(semantic_tags={"numeric"})]
    return_type = ColumnSchema(logical_type=IntegerNullable, semantic_tags={"numeric"})
    stack_on_self = False

    def __init__(self, skipna=True):
        self.skipna = skipna

    def get_function(self):
        def count_below_mean(x):
            mean = x.mean(skipna=self.skipna)
            if np.isnan(mean):
                return np.nan
            return len(x[x < mean])

        return count_below_mean
