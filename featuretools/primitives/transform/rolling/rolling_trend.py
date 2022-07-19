import numpy as np
import pandas as pd
from woodwork.column_schema import ColumnSchema
from woodwork.logical_types import Datetime, Double

from featuretools.primitives.core.transform_primitive import TransformPrimitive
from featuretools.primitives.rolling_primitive_utils import (
    apply_roll_with_offset_gap,
    roll_series_with_gap,
)
from featuretools.utils import calculate_trend

class RollingTrend(TransformPrimitive):
    """Calculates the trend of a given window of entries of a column over time.

    Description:
        Given a list of numbers and a corresponding list of
        datetimes, return a rolling slope of the linear trend
        of values, starting at the row `gap` rows away from the current row and looking backward
        over the specified time window (by `window_length` and `gap`).

        Input datetimes should be monotonic.

     Args:
        window_length (int, string, optional): Specifies the amount of data included in each window.
            If an integer is provided, will correspond to a number of rows. For data with a uniform sampling frequency,
            for example of one day, the window_length will correspond to a period of time, in this case,
            7 days for a window_length of 7.
            If a string is provided, it must be one of pandas' offset alias strings ('1D', '1H', etc),
            and it will indicate a length of time that each window should span.
            The list of available offset aliases, can be found at
            https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases.
            Defaults to 3.
        gap (int, string, optional): Specifies a gap backwards from each instance before the
            window of usable data begins. If an integer is provided, will correspond to a number of rows.
            If a string is provided, it must be one of pandas' offset alias strings ('1D', '1H', etc),
            and it will indicate a length of time between a target instance and the beginning of its window.
            Defaults to 0, which will include the target instance in the window.
        min_periods (int, optional): Minimum number of observations required for performing calculations
            over the window. Can only be as large as window_length when window_length is an integer.
            When window_length is an offset alias string, this limitation does not exist, but care should be taken
            to not choose a min_periods that will always be larger than the number of observations in a window.
            Defaults to 1.

    Examples:
        >>> import pandas as pd
        >>> rolling_trend = RollingTrend()
        >>> times = pd.date_range(start="2019-01-01", freq="1D", periods=10)
        >>> rolling_trend(times, [1, 2, 4, 8, 16, 24, 48, 96, 192, 384]).tolist()
        [nan, nan, 1.4999999999999998, 2.9999999999999996, 5.999999999999999, 7.999999999999999, 16.0, 36.0, 72.0, 144.0]

        We can also control the gap before the rolling calculation.

        >>> rolling_trend = RollingTrend(gap=1)
        >>> rolling_trend(times, [1, 2, 4, 8, 16, 24, 48, 96, 192, 384]).tolist()
        [nan, nan, nan, 1.4999999999999998, 2.9999999999999996, 5.999999999999999, 7.999999999999999, 16.0, 36.0, 72.0]

        We can also control the minimum number of periods required for the rolling calculation.

        >>> rolling_trend = RollingTrend(window_length=4, min_periods=4)
        >>> rolling_trend(times, [1, 2, 4, 8, 16, 24, 48, 96, 192, 384]).tolist()
        [nan, nan, nan, 2.299999999999999, 4.599999999999998, 6.799999999999996, 12.799999999999992, 26.399999999999984, 55.19999999999997, 110.39999999999993]

        We can also set the window_length and gap using offset alias strings.

        >>> rolling_trend = RollingTrend(window_length="4D", gap="1D")
        >>> rolling_trend(times, [1, 2, 4, 8, 16, 24, 48, 96, 192, 384]).tolist()
        [nan, nan, nan, 1.4999999999999998, 2.299999999999999, 4.599999999999998, 6.799999999999996, 12.799999999999992, 26.399999999999984, 55.19999999999997]
    """

    name = "rolling_trend"
    input_types = [
        ColumnSchema(logical_type=Datetime, semantic_tags={"time_index"}),
        ColumnSchema(semantic_tags={"numeric"}),
    ]
    return_type = ColumnSchema(logical_type=Double, semantic_tags={"numeric"})

    def __init__(self, window_length=3, gap=0, min_periods=0):
        self.window_length = window_length
        self.gap = gap
        self.min_periods = min_periods

    def get_function(self):
        def rolling_trend(datetime, numeric):
            x = pd.Series(numeric.values, index=datetime.values)
            rolled_series = roll_series_with_gap(
                x, self.window_length, gap=self.gap, min_periods=self.min_periods
            )
            if isinstance(self.gap, str):
                additional_args = (self.gap, calculate_trend, self.min_periods)
                return rolled_series.apply(
                    apply_roll_with_offset_gap,
                    args=additional_args,
                ).values
            return rolled_series.apply(calculate_trend).values

        return rolling_trend