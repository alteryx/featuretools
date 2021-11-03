import pandas as pd

from woodwork.column_schema import ColumnSchema
from woodwork.logical_types import (
    Double,
    Datetime
)
from featuretools.primitives.utils import roll_series_with_gap
from featuretools.primitives.base.transform_primitive_base import (
    TransformPrimitive
)


class RollingMax(TransformPrimitive):
    # --> maybe put in its own file? either a time series or a rolling file
    """Determines the maximum of entries over a given timeframe.

    Description:
        Given a list of numbers and a corresponding list of
        datetimes, return a rolling maximum of the numeric values,
        starting at the current row and looking backward
        over the specified time window (`time_frame`).

        Input datetimes should be monotonic.

    Args:
    # --> update
        time_frame (str): The time period of each frame. Time frames
            should be in seconds, minutes, hours, or days (e.g. 1s,
            5min, 4h, 7d, etc.). Defaults to 1 day.

    Examples:
        >>> import pandas as pd
        >>> rolling_max = RollingMax()
        >>> times = pd.date_range(start='2019-01-01', freq='1min', periods=5)
        >>> rolling_max(times, [4, 3, 2, 1, 0]).tolist()
        [4.0, 4.0, 4.0, 4.0, 4.0]

        We can control the time frame of the rolling calculation.

        >>> import pandas as pd
        >>> rolling_max = RollingMax(time_frame='2min')
        >>> times = pd.date_range(start='2019-01-01', freq='1min', periods=5)
        >>> rolling_max(times, [4, 3, 2, 1, 0]).tolist()
        [4.0, 4.0, 3.0, 2.0, 1.0]
    """
    name = "rolling_max"
    input_types = [ColumnSchema(logical_type=Datetime, semantic_tags={'time_index'}), ColumnSchema(semantic_tags={'numeric'})]
    return_type = ColumnSchema(logical_type=Double, semantic_tags={'numeric'})

    def __init__(self, window_length=1, gap=0, min_periods=0):
        self.window_length = window_length
        self.gap = gap
        self.min_periods = min_periods
        # -->determine if we need to add the uses_full_dataframe = True right now

    def get_function(self):
        def rolling_max(datetime, numeric):
            x = pd.Series(numeric.values, index=datetime.values)
            rolled_series = roll_series_with_gap(x,
                                                 self.window_length,
                                                 gap=self.gap,
                                                 min_periods=self.min_periods)
            return rolled_series.max().values
        return rolling_max
