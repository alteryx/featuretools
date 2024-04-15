import pandas as pd
from woodwork.column_schema import ColumnSchema
from woodwork.logical_types import Boolean, BooleanNullable, Double

from featuretools.primitives.base.aggregation_primitive_base import AggregationPrimitive


class PercentTrue(AggregationPrimitive):
    """Determines the percent of `True` values.

    Description:
        Given a list of booleans, return the percent
        of values which are `True` as a decimal.
        `NaN` values are treated as `False`,
        adding to the denominator.

    Examples:
        >>> percent_true = PercentTrue()
        >>> percent_true([True, False, True, True, None])
        0.6
    """

    name = "percent_true"
    input_types = [
        [ColumnSchema(logical_type=BooleanNullable)],
        [ColumnSchema(logical_type=Boolean)],
    ]
    return_type = ColumnSchema(logical_type=Double, semantic_tags={"numeric"})
    stack_on = []
    stack_on_exclude = []
    default_value = pd.NA
    description_template = "the percentage of true values in {}"

    def get_function(self):
        def percent_true(s):
            return s.fillna(False).mean()

        return percent_true
