import pandas as pd
from woodwork.column_schema import ColumnSchema
from woodwork.logical_types import Boolean, BooleanNullable

from featuretools.primitives.base import TransformPrimitive


class Lag(TransformPrimitive):
    """Shifts an array of values by a specified number of periods.

    Args:
        periods (int): The number of periods by which to shift the input.
            Default is 1. Periods correspond to rows.

    Examples:
        >>> lag = Lag()
        >>> lag([1, 2, 3, 4, 5], pd.Series(pd.date_range(start="2020-01-01", periods=5, freq='D'))).tolist()
        [nan, 1.0, 2.0, 3.0, 4.0]

        You can specify the number of periods to shift the values

        >>> lag_periods = Lag(periods=3)
        >>> lag_periods([True, False, False, True, True], pd.Series(pd.date_range(start="2020-01-01", periods=5, freq='D'))).tolist()
        [nan, nan, nan, True, False]
    """

    # Note: with pandas 1.5.0, using Lag with a string input will result in `None` values
    # being introduced instead of `nan` values that were present in previous versions.
    # All missing values will be replaced by `np.nan` (for Double) or `pd.NA` (all other types)
    # once Woodwork is initialized on the feature matrix.
    name = "lag"
    input_types = [
        [
            ColumnSchema(semantic_tags={"category"}),
            ColumnSchema(semantic_tags={"time_index"}),
        ],
        [
            ColumnSchema(semantic_tags={"numeric"}),
            ColumnSchema(semantic_tags={"time_index"}),
        ],
        [
            ColumnSchema(logical_type=Boolean),
            ColumnSchema(semantic_tags={"time_index"}),
        ],
        [
            ColumnSchema(logical_type=BooleanNullable),
            ColumnSchema(semantic_tags={"time_index"}),
        ],
    ]
    return_type = None
    uses_full_dataframe = True

    def __init__(self, periods=1):
        self.periods = periods

    def get_function(self):
        def lag(input_col, time_index):
            x = pd.Series(input_col.values, index=time_index.values)
            return x.shift(periods=self.periods, fill_value=None).values

        return lag
