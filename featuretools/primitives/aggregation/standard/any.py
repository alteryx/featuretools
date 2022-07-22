import numpy as np
from dask import dataframe as dd
from woodwork.column_schema import ColumnSchema
from woodwork.logical_types import Boolean, BooleanNullable

from featuretools.primitives.core.aggregation_primitive import AggregationPrimitive
from featuretools.utils.gen_utils import Library


class Any(AggregationPrimitive):
    """Determines if any value is 'True' in a list.

    Description:
        Given a list of booleans, return `True` if one or
        more of the values are `True`.

    Examples:
        >>> any = Any()
        >>> any([False, False, False, True])
        True
    """

    name = "any"
    input_types = [
        [ColumnSchema(logical_type=Boolean)],
        [ColumnSchema(logical_type=BooleanNullable)],
    ]
    return_type = ColumnSchema(logical_type=Boolean)
    stack_on_self = False
    compatibility = [Library.PANDAS, Library.DASK]
    description_template = "whether any of {} are true"

    def get_function(self, agg_type=Library.PANDAS):
        if agg_type == Library.DASK:

            def chunk(s):
                return s.agg(np.any)

            def agg(s):
                return s.agg(np.any)

            return dd.Aggregation(self.name, chunk=chunk, agg=agg)

        return np.any