import numpy as np
from woodwork.column_schema import ColumnSchema
from woodwork.logical_types import Integer

from featuretools.primitives.base import AggregationPrimitive


class NumZeroCrossings(AggregationPrimitive):
    """Determines the number of times a list crosses 0.
    Description:
        Given a list of numbers, return the number of times the value
        crosses 0. It is the number of times the value goes from a
        positive number to a negative number, or a negative number to
        a positive number. NaN values are ignored.
    Examples:
        >>> num_zero_crossings = NumZeroCrossings()
        >>> num_zero_crossings([1, -1, 2, -2, 3, -3])
        5
    """

    name = "num_zero_crossings"
    input_types = [ColumnSchema(semantic_tags={"numeric"})]
    return_type = ColumnSchema(logical_type=Integer, semantic_tags={"numeric"})

    def get_function(self):
        def num_zero_crossings(x):
            cleaned = x[(x != 0) & (x == x)]
            signs = np.sign(cleaned)
            difference = np.diff(signs)
            crossings = np.where(difference)[0]
            return len(crossings)

        return num_zero_crossings
