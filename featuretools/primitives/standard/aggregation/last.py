from woodwork.column_schema import ColumnSchema

from featuretools.primitives.base.aggregation_primitive_base import AggregationPrimitive


class Last(AggregationPrimitive):
    """Determines the last value in a list.

    Examples:
        >>> last = Last()
        >>> last([1, 2, 3, 4, 5, None])
        nan
    """

    name = "last"
    input_types = [ColumnSchema()]
    return_type = None
    stack_on_self = False
    description_template = "the last instance of {}"

    def get_function(self):
        def pd_last(x):
            return x.iloc[-1]

        return pd_last
