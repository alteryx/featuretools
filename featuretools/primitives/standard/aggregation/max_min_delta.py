from woodwork.column_schema import ColumnSchema

from featuretools.primitives.base import AggregationPrimitive


class MaxMinDelta(AggregationPrimitive):
    """Determines the difference between the max and min value.

    Args:
        skipna (bool): Determines if to use NA/null values.
            Defaults to True to skip NA/null.

    Examples:
        >>> max_min_delta = MaxMinDelta()
        >>> max_min_delta([7, 2, 5, 3, 10])
        8

        You can optionally specify how to handle NaN values

        >>> max_min_delta_skipna = MaxMinDelta(skipna=False)
        >>> max_min_delta_skipna([7, 2, None, 3, 10])
        nan
    """

    name = "max_min_delta"
    input_types = [ColumnSchema(semantic_tags={"numeric"})]
    return_type = ColumnSchema(semantic_tags={"numeric"})
    stack_on_self = False
    default_value = 0

    def __init__(self, skipna=True):
        self.skipna = skipna

    def get_function(self):
        def max_min_delta(x):
            max_val = x.max(skipna=self.skipna)
            min_val = x.min(skipna=self.skipna)
            return max_val - min_val

        return max_min_delta
