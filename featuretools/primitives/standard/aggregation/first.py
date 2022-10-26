from woodwork.column_schema import ColumnSchema

from featuretools.primitives.base.aggregation_primitive_base import AggregationPrimitive
from featuretools.utils.gen_utils import Library


class First(AggregationPrimitive):
    """Determines the first value in a list.

    Examples:
        >>> first = First()
        >>> first([1, 2, 3, 4, 5, None])
        1.0
    """

    name = "first"
    input_types = [ColumnSchema()]
    return_type = None
    stack_on_self = False
    description_template = "the first instance of {}"

    def get_function(self, agg_type=Library.PANDAS):
        def pd_first(x):
            return x.iloc[0]

        return pd_first
