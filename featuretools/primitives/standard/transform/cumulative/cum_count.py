import numpy as np
from woodwork.column_schema import ColumnSchema
from woodwork.logical_types import IntegerNullable

from featuretools.primitives.base import TransformPrimitive


class CumCount(TransformPrimitive):
    """Calculates the cumulative count.

    Description:
        Given a list of values, return the cumulative count
        (or running count). There is no set window, so the
        count at each point is calculated over all prior
        values. `NaN` values are counted.

    Examples:
        >>> cum_count = CumCount()
        >>> cum_count([1, 2, 3, 4, None, 5]).tolist()
        [1, 2, 3, 4, 5, 6]
    """

    name = "cum_count"
    input_types = [
        [ColumnSchema(semantic_tags={"foreign_key"})],
        [ColumnSchema(semantic_tags={"category"})],
    ]
    return_type = ColumnSchema(logical_type=IntegerNullable, semantic_tags={"numeric"})
    uses_full_dataframe = True
    description_template = "the cumulative count of {}"

    def get_function(self):
        def cum_count(values):
            return np.arange(1, len(values) + 1)

        return cum_count
