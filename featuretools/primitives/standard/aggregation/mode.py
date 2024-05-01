import numpy as np
from woodwork.column_schema import ColumnSchema

from featuretools.primitives.base.aggregation_primitive_base import AggregationPrimitive


class Mode(AggregationPrimitive):
    """Determines the most commonly repeated value.

    Description:
        Given a list of values, return the value with the
        highest number of occurences. If list is
        empty, return `NaN`.

    Examples:
        >>> mode = Mode()
        >>> mode(['red', 'blue', 'green', 'blue'])
        'blue'
    """

    name = "mode"
    input_types = [ColumnSchema(semantic_tags={"category"})]
    return_type = None
    description_template = "the most frequently occurring value of {}"

    def get_function(self):
        def pd_mode(s):
            return s.mode().get(0, np.nan)

        return pd_mode
