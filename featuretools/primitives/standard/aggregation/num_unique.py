import pandas as pd
from woodwork.column_schema import ColumnSchema
from woodwork.logical_types import IntegerNullable

from featuretools.primitives.base.aggregation_primitive_base import AggregationPrimitive


class NumUnique(AggregationPrimitive):
    """Determines the number of distinct values, ignoring `NaN` values.

    Args:
        use_string_for_pd_calc (bool): Determines if the string 'nunique' or the function
            pd.Series.nunique is used for making the primitive calculation. Put in place to
            account for the bug https://github.com/pandas-dev/pandas/issues/57317.
            Defaults to using the string.

    Examples:
        >>> num_unique = NumUnique(use_string_for_pd_calc=False)
        >>> num_unique(['red', 'blue', 'green', 'yellow'])
        4

        `NaN` values will be ignored.

        >>> num_unique(['red', 'blue', 'green', 'yellow', None])
        4
    """

    name = "num_unique"
    input_types = [ColumnSchema(semantic_tags={"category"})]
    return_type = ColumnSchema(logical_type=IntegerNullable, semantic_tags={"numeric"})
    stack_on_self = False
    description_template = "the number of unique elements in {}"

    def __init__(self, use_string_for_pd_calc=True):
        self.use_string_for_pd_calc = use_string_for_pd_calc

    def get_function(self):
        if self.use_string_for_pd_calc:
            return "nunique"
        return pd.Series.nunique
