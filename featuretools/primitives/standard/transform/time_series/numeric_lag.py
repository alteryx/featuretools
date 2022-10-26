import warnings

import pandas as pd
from woodwork.column_schema import ColumnSchema

from featuretools.primitives.base import TransformPrimitive


class NumericLag(TransformPrimitive):
    """Shifts an array of values by a specified number of periods.

    Args:
        periods (int): The number of periods by which to shift the input.
            Default is 1. Periods correspond to rows.

        fill_value (int, float, optional): The value to use to fill in
            the gaps left after shifting the input. Default is None.

    Examples:
        >>> lag = NumericLag()
        >>> lag(pd.Series(pd.date_range(start="2020-01-01", periods=5, freq='D')), [1, 2, 3, 4, 5]).tolist()
        [nan, 1.0, 2.0, 3.0, 4.0]

        You can specify the number of periods to shift the values

        >>> lag_periods = NumericLag(periods=3)
        >>> lag_periods(pd.Series(pd.date_range(start="2020-01-01", periods=5, freq='D')), [1, 2, 3, 4, 5]).tolist()
        [nan, nan, nan, 1.0, 2.0]

        You can specify the fill value to use

        >>> lag_fill_value = NumericLag(fill_value=100)
        >>> lag_fill_value(pd.Series(pd.date_range(start="2020-01-01", periods=4, freq='D')), [1, 2, 3, 4]).tolist()
        [100, 1, 2, 3]
    """

    name = "numeric_lag"
    input_types = [
        ColumnSchema(semantic_tags={"time_index"}),
        ColumnSchema(semantic_tags={"numeric"}),
    ]
    return_type = ColumnSchema(semantic_tags={"numeric"})
    uses_full_dataframe = True

    def __init__(self, periods=1, fill_value=None):
        self.periods = periods
        self.fill_value = fill_value
        warnings.warn(
            "NumericLag is deprecated and will be removed in a future version. Please use the 'Lag' primitive instead.",
            FutureWarning,
        )

    def get_function(self):
        def lag(time_index, numeric):
            x = pd.Series(numeric.values, index=time_index.values)
            return x.shift(periods=self.periods, fill_value=self.fill_value).values

        return lag
