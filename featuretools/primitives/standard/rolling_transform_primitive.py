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
        #     --> confirm default window length - 1 seems possibly a bad idea bc youre not getting windows
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


class RollingMin(TransformPrimitive):
    """Determines the minimum of entries over a given timeframe.

    Description:
        Given a list of numbers and a corresponding list of
        datetimes, return a rolling minimum of the numeric values,
        starting at the current row and looking backward
        over the specified time window (`time_frame`).
        Input datetimes should be monotonic.

    Args:
        time_frame (str): The time period of each frame. Time frames
            should be in seconds, minutes, hours, or days (e.g. 1s,
            5min, 4h, 7d, etc.). Defaults to 1 day.

    Examples:
        >>> import pandas as pd
        >>> times = pd.date_range(start='2019-01-01', freq='1min', periods=5)
        >>> rolling_min = RollingMin()
        >>> rolling_min(times, [0, 1, 2, 3, 4]).tolist()
        [0.0, 0.0, 0.0, 0.0, 0.0]

        We can control the time frame of the rolling calculation.

        >>> import pandas as pd
        >>> rolling_min = RollingMin(time_frame='2min')
        >>> rolling_min(times, [0, 1, 2, 3, 4]).tolist()
        [0.0, 0.0, 1.0, 2.0, 3.0]
    """
    name = "rolling_min"
    input_types = [ColumnSchema(logical_type=Datetime, semantic_tags={'time_index'}), ColumnSchema(semantic_tags={'numeric'})]
    return_type = ColumnSchema(logical_type=Double, semantic_tags={'numeric'})

    def __init__(self, window_length=1, gap=0, min_periods=0):
        self.window_length = window_length
        self.gap = gap
        self.min_periods = min_periods

    def get_function(self):
        def rolling_min(datetime, numeric):
            x = pd.Series(numeric.values, index=datetime.values)
            rolled_series = roll_series_with_gap(x,
                                                 self.window_length,
                                                 gap=self.gap,
                                                 min_periods=self.min_periods)
            return rolled_series.min().values
        return rolling_min


class RollingMean(TransformPrimitive):
    """Calculates the mean of entries over a given timeframe.

    Description:
        Given a list of numbers and a corresponding list of
        datetimes, return a rolling mean of the numeric values,
        starting at the current row and looking backward
        over the specified time window (`time_frame`).

        Input datetimes should be monotonic.

    Args:
        time_frame (str): The time period of each frame. Time frames
            should be in seconds, minutes, hours, or days (e.g. 1s,
            5min, 4h, 7d, etc.). Defaults to 1 day.

    Examples:
        >>> import pandas as pd
        >>> rolling_mean = RollingMean()
        >>> times = pd.date_range(start='2019-01-01', freq='1min', periods=5)
        >>> rolling_mean(times, [4, 3, 2, 1, 0]).tolist()
        [4.0, 3.5, 3.0, 2.5, 2.0]

        We can control the time frame of the rolling calculation.

        >>> import pandas as pd
        >>> rolling_mean = RollingMean(time_frame='2min')
        >>> times = pd.date_range(start='2019-01-01', freq='1min', periods=5)
        >>> rolling_mean(times, [4, 3, 2, 1, 0]).tolist()
        [4.0, 3.5, 2.5, 1.5, 0.5]
    """
    name = "rolling_mean"
    input_types = [ColumnSchema(logical_type=Datetime, semantic_tags={'time_index'}), ColumnSchema(semantic_tags={'numeric'})]
    return_type = ColumnSchema(logical_type=Double, semantic_tags={'numeric'})

    def __init__(self, window_length=1, gap=0, min_periods=0):
        self.window_length = window_length
        self.gap = gap
        self.min_periods = min_periods

    def get_function(self):
        def rolling_mean(datetime, numeric):
            x = pd.Series(numeric.values, index=datetime.values)
            rolled_series = roll_series_with_gap(x,
                                                 self.window_length,
                                                 gap=self.gap,
                                                 min_periods=self.min_periods)
            return rolled_series.mean().values
        return rolling_mean


class RollingSTD(TransformPrimitive):
    """Calculates the standard deviation of entries over a given timeframe.

    Description:
        Given a list of numbers and a corresponding list of
        datetimes, return a rolling standard deviation of
        the numeric values, starting at the current row and
        looking backward over the specified time window
        (`time_frame`). Input datetimes should be monotonic.

    Args:
        time_frame (str): The time period of each frame. Time frames
            should be in seconds, minutes, hours, or days (e.g. 1s,
            5min, 4h, 7d, etc.). Defaults to 1 day.

    Examples:
        >>> import pandas as pd
        >>> times = pd.date_range(start='2019-01-01', freq='1min', periods=5)
        >>> rolling_std = RollingSTD()
        >>> rolling_std(times, [1, 7, 3, 9, 5]).tolist()
        [nan, 4.242640687119285, 3.0550504633038935, 3.6514837167011076, 3.1622776601683795]

        We can control the time frame of the rolling calculation.

        >>> import pandas as pd
        >>> rolling_std = RollingSTD(time_frame='2min')
        >>> rolling_std(times, [1, 7, 3, 9, 5]).tolist()
        [nan, 4.242640687119285, 2.8284271247461903, 4.242640687119285, 2.8284271247461903]
    """
    name = "rolling_std"
    input_types = [ColumnSchema(logical_type=Datetime, semantic_tags={'time_index'}), ColumnSchema(semantic_tags={'numeric'})]
    return_type = ColumnSchema(logical_type=Double, semantic_tags={'numeric'})

    def __init__(self, time_frame='1d'):
        try:
            pd.tseries.frequencies.to_offset(time_frame).nanos
            self.time_frame = time_frame
        except Exception:
            raise ValueError('must be a valid time frame in seconds, minutes, hours, or days (e.g. 1s, 5min, 4h, 7d, etc.)')

    def get_function(self):
        def rolling_std(datetime, numeric):
            x = pd.Series(numeric.values, index=datetime.values)
            return x.rolling(self.time_frame).std().values
        return rolling_std

    def __init__(self, window_length=1, gap=0, min_periods=0):
        self.window_length = window_length
        self.gap = gap
        self.min_periods = min_periods

    def get_function(self):
        def rolling_std(datetime, numeric):
            x = pd.Series(numeric.values, index=datetime.values)
            rolled_series = roll_series_with_gap(x,
                                                 self.window_length,
                                                 gap=self.gap,
                                                 min_periods=self.min_periods)
            return rolled_series.std().values
        return rolling_std
