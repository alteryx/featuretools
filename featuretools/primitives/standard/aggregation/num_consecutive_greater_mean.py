import numpy as np
from woodwork.column_schema import ColumnSchema
from woodwork.logical_types import IntegerNullable

from featuretools.primitives.base import AggregationPrimitive


class NumConsecutiveGreaterMean(AggregationPrimitive):
    """Determines the length of the longest subsequence above the mean.

    Description:
        Given a list of numbers, find the longest subsequence of numbers
        larger than the mean of the entire sequence. Return the length
        of the longest subsequence.

    Args:
        skipna (bool): If this is False and any value in x is `NaN`, then
            the result will be `NaN`. If True, `NaN` values are skipped.
            Default is True.

    Examples:
        >>> num_consecutive_greater_mean = NumConsecutiveGreaterMean()
        >>> num_consecutive_greater_mean([1, 2, 3, 4, 5, 6])
        3.0

        We can also control the way `NaN` values are handled.

        >>> num_consecutive_greater_mean = NumConsecutiveGreaterMean(skipna=False)
        >>> num_consecutive_greater_mean([1, 2, 3, 4, 5, 6, None])
        nan
    """

    name = "num_consecutive_greater_mean"
    input_types = [ColumnSchema(semantic_tags={"numeric"})]
    return_type = ColumnSchema(logical_type=IntegerNullable, semantic_tags={"numeric"})
    stack_on_self = False
    default_value = 0

    def __init__(self, skipna=True):
        self.skipna = skipna

    def get_function(self):
        def num_consecutive_greater_mean(x):
            # check for NaN cases
            if x.isnull().all():
                return np.nan
            if not self.skipna and x.isnull().values.any():
                return np.nan
            x_mean = x.mean()

            # In some cases, the mean of x may be NaN
            #   (such as when x has both inf and -inf values)
            if np.isnan(x.mean()):
                return np.nan

            # Find indices of points at or below mean
            x = x.dropna().reset_index(drop=True)
            below_mean_indices = x[x <= x_mean].index.to_series()

            # If none of x is below the mean, return the length of x
            if below_mean_indices.empty:
                return len(x)

            # Pad index with start/end values, in case the longest
            #   sequence occurs at the beginning or end of x
            below_mean_indices[-1] = -1
            below_mean_indices[len(x)] = len(x)
            below_mean_indices = below_mean_indices.sort_index()

            # Calculate gaps between points below mean
            below_mean_indices_shifted = below_mean_indices.shift(1)
            diffs = below_mean_indices - below_mean_indices_shifted

            # Take biggest gap, and subtract 1 to get result
            max_gap = (diffs).max() - 1
            return max_gap

        return num_consecutive_greater_mean
