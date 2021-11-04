import numpy as np
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
from featuretools.utils.gen_utils import Library


class RollingMax(TransformPrimitive):
    """Determines the maximum of entries over a given window.

    Description:
        Given a list of numbers and a corresponding list of
        datetimes, return a rolling maximum of the numeric values,
        starting at the row `gap` rows away from the current row and looking backward
        over the specified window (by `window_length` and `gap`). 

        Input datetimes should be monotonic.

    Args:
        window_length (int): The number of rows to be included in each frame. For data
            with a uniform sampling frequency, for example of one day, the window_length will
            correspond to a period of time, in this case, 7 days for a window_length of 7.
        gap (int, optional): The number of rows prior to each instance to be skipped before
            beginning the window over which the maximum is determined. Defaults to 0, which
            will include each instance in the window.
        min_periods (int, optional): Minimum number of observations required for a window to have a value.
            Can only be as large as window_length. Defaults to 1.

    Examples:
        >>> import pandas as pd
        >>> rolling_max = RollingMax(window_length=3)
        >>> times = pd.date_range(start='2019-01-01', freq='1min', periods=5)
        >>> rolling_max(times, [4, 3, 2, 1, 0]).tolist()
        [4.0, 4.0, 4.0, 3.0, 2.0]

        We can also control the gap before the rolling calculation.

        >>> import pandas as pd
        >>> rolling_max = RollingMax(window_length=3, gap=1)
        >>> times = pd.date_range(start='2019-01-01', freq='1min', periods=5)
        >>> rolling_max(times, [4, 3, 2, 1, 0]).tolist()
        [NaN, 4.0, 4.0, 3.0, 2.0]

        We can also control the minimum number of periods required for the rolling calculation.

        >>> import pandas as pd
        >>> rolling_max = RollingMax(window_length=3, min_periods=3)
        >>> times = pd.date_range(start='2019-01-01', freq='1min', periods=5)
        >>> rolling_max(times, [4, 3, 2, 1, 0]).tolist()
        [NaN, NaN, 4.0, 3.0, 2.0]
    """
    name = "rolling_max"
    input_types = [ColumnSchema(logical_type=Datetime, semantic_tags={'time_index'}), ColumnSchema(semantic_tags={'numeric'})]
    return_type = ColumnSchema(logical_type=Double, semantic_tags={'numeric'})
    compatibility = [Library.PANDAS, Library.DASK, Library.KOALAS]

    def __init__(self, window_length, gap=0, min_periods=1):
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
    """Determines the minimum of entries over a given window.

    Description:
        Given a list of numbers and a corresponding list of
        datetimes, return a rolling minimum of the numeric values,
        starting at the row `gap` rows away from the current row and looking backward
        over the specified window (by `window_length` and `gap`).
        Input datetimes should be monotonic.

    Args:
        window_length (int): The number of rows to be included in each frame. For data
            with a uniform sampling frequency, for example of one day, the window_length will
            correspond to a period of time, in this case, 7 days for a window_length of 7.
        gap (int, optional): The number of rows prior to each instance to be skipped before
            beginning the window over which the minimum is determined. Defaults to 0, which
            will include each instance in the window.
        min_periods (int, optional): Minimum number of observations required for a window to have a value.
            Defaults to 1.
    Examples:
        >>> import pandas as pd
        >>> rolling_min = RollingMin(window_length=3)
        >>> times = pd.date_range(start='2019-01-01', freq='1min', periods=5)
        >>> rolling_min(times, [4, 3, 2, 1, 0]).tolist()
        [4.0, 3.0, 2.0, 1.0, 0.0]

        We can also control the gap before the rolling calculation.

        >>> import pandas as pd
        >>> rolling_min = RollingMin(window_length=3, gap=1)
        >>> times = pd.date_range(start='2019-01-01', freq='1min', periods=5)
        >>> rolling_min(times, [4, 3, 2, 1, 0]).tolist()
        [NaN, 3.0, 2.0, 1.0, 0.0]

        We can also control the minimum number of periods required for the rolling calculation.

        >>> import pandas as pd
        >>> rolling_min = RollingMin(window_length=3, min_periods=3)
        >>> times = pd.date_range(start='2019-01-01', freq='1min', periods=5)
        >>> rolling_min(times, [4, 3, 2, 1, 0]).tolist()
        [NaN, NaN, 2.0, 1.0, 0.0]
    """
    name = "rolling_min"
    input_types = [ColumnSchema(logical_type=Datetime, semantic_tags={'time_index'}), ColumnSchema(semantic_tags={'numeric'})]
    return_type = ColumnSchema(logical_type=Double, semantic_tags={'numeric'})
    compatibility = [Library.PANDAS, Library.DASK, Library.KOALAS]

    def __init__(self, window_length, gap=0, min_periods=1):
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
        starting at the row `gap` rows away from the current row and looking backward
        over the specified time window (by `window_length` and `gap`).

        Input datetimes should be monotonic.

    Args:
        window_length (int): The number of rows to be included in each frame. For data
            with a uniform sampling frequency, for example of one day, the window_length will
            correspond to a period of time, in this case, 7 days for a window_length of 7.
        gap (int, optional): The number of rows prior to each instance to be skipped before
            beginning the window over which the mean is determined. Defaults to 0, which
            will include each instance in the window.
        min_periods (int, optional): Minimum number of observations required for a window to have a value.
            Can only be as large as window_length. Defaults to 1.

    Examples:
        >>> import pandas as pd
        >>> rolling_mean = RollingMean(window_length=3)
        >>> times = pd.date_range(start='2019-01-01', freq='1min', periods=5)
        >>> rolling_mean(times, [4, 3, 2, 1, 0]).tolist()
        [4.0, 3.5, 3.0, 2.0, 1.0]

        We can also control the gap before the rolling calculation.

        >>> import pandas as pd
        >>> rolling_mean = RollingMean(window_length=3, gap=1)
        >>> times = pd.date_range(start='2019-01-01', freq='1min', periods=5)
        >>> rolling_mean(times, [4, 3, 2, 1, 0]).tolist()
        [NaN, 4.0, 3.5, 3.0, 2.0]

        We can also control the minimum number of periods required for the rolling calculation.

        >>> import pandas as pd
        >>> rolling_mean = RollingMean(window_length=3, min_periods=3)
        >>> times = pd.date_range(start='2019-01-01', freq='1min', periods=5)
        >>> rolling_mean(times, [4, 3, 2, 1, 0]).tolist()
        [NaN, NaN, 3.0, 2.0, 1.0]
    """
    name = "rolling_mean"
    input_types = [ColumnSchema(logical_type=Datetime, semantic_tags={'time_index'}), ColumnSchema(semantic_tags={'numeric'})]
    return_type = ColumnSchema(logical_type=Double, semantic_tags={'numeric'})
    compatibility = [Library.PANDAS, Library.DASK, Library.KOALAS]

    def __init__(self, window_length, gap=0, min_periods=0):
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
        the numeric values, starting at the row `gap` rows away from the current row and
        looking backward over the specified time window 
        (by `window_length` and `gap`). Input datetimes should be monotonic.

    Args:
        window_length (int): The number of rows to be included in each frame. For data
            with a uniform sampling frequency, for example of one day, the window_length will
            correspond to a period of time, in this case, 7 days for a window_length of 7.
        gap (int, optional): The number of rows prior to each instance to be skipped before
            beginning the window over which the standard deviation is determined. Defaults to 0, which
            will include each instance in the window.
        min_periods (int, optional): Minimum number of observations required for a window to have a value.
            Can only be as large as window_length. Defaults to 1.

    Examples:
        >>> import pandas as pd
        >>> rolling_std = RollingSTD(window_length=4)
        >>> times = pd.date_range(start='2019-01-01', freq='1min', periods=5)
        >>> rolling_std(times, [4, 3, 2, 1, 0]).tolist()
        [nan, 0.7071067811865476, 1.0, 1.2909944487358056, 1.2909944487358056]

        We can also control the gap before the rolling calculation.

        >>> import pandas as pd
        >>> rolling_std = RollingSTD(window_length=4, gap=1)
        >>> times = pd.date_range(start='2019-01-01', freq='1min', periods=5)
        >>> rolling_std(times, [4, 3, 2, 1, 0]).tolist()
        [nan, nan, 0.7071067811865476, 1.0, 1.2909944487358056]

        We can also control the minimum number of periods required for the rolling calculation.

        >>> import pandas as pd
        >>> rolling_std = RollingSTD(window_length=4, min_periods=4)
        >>> times = pd.date_range(start='2019-01-01', freq='1min', periods=5)
        >>> rolling_std(times, [4, 3, 2, 1, 0]).tolist()
        [nan, nan, nan, 1.2909944487358056, 1.2909944487358056]
    """
    name = "rolling_std"
    input_types = [ColumnSchema(logical_type=Datetime, semantic_tags={'time_index'}), ColumnSchema(semantic_tags={'numeric'})]
    return_type = ColumnSchema(logical_type=Double, semantic_tags={'numeric'})
    compatibility = [Library.PANDAS, Library.DASK, Library.KOALAS]

    def __init__(self, window_length, gap=0, min_periods=1):
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


class RollingCount(TransformPrimitive):
    """Determines a rolling count of events over a given timeframe.

    Description:
        Given a list of datetimes, return a rolling count starting
        at the row `gap` rows away from the current row and looking backward over the specified
        time window (by `window_length` and `gap`).

        Input datetimes should be monotonic.

    Args:
        window_length (int): The number of rows to be included in each frame. For data
            with a uniform sampling frequency, for example of one day, the window_length will
            correspond to a period of time, in this case, 7 days for a window_length of 7.
        gap (int, optional): The number of rows prior to each instance to be skipped before
            beginning the window over which the count is determined. Defaults to 0, which
            will include each instance in the window.
        min_periods (int, optional): Minimum number of observations required for a window to have a value.
            Can only be as large as window_length. Defaults to 1.

    Examples:
        >>> import pandas as pd
        >>> rolling_count = RollingCount(window_length=3)
        >>> times = pd.date_range(start='2019-01-01', freq='1min', periods=5)
        >>> rolling_count(times, [4, 3, 2, 1, 0]).tolist()
        [1.0, 2.0, 3.0, 3.0, 3.0]

        We can also control the gap before the rolling calculation.

        >>> import pandas as pd
        >>> rolling_count = RollingCount(window_length=3, gap=1)
        >>> times = pd.date_range(start='2019-01-01', freq='1min', periods=5)
        >>> rolling_count(times, [4, 3, 2, 1, 0]).tolist()
        [NaN, 1.0, 2.0, 3.0, 3.0]

        We can also control the minimum number of periods required for the rolling calculation.

        >>> import pandas as pd
        >>> rolling_count = RollingCount(window_length=3, min_periods=3)
        >>> times = pd.date_range(start='2019-01-01', freq='1min', periods=5)
        >>> rolling_count(times, [4, 3, 2, 1, 0]).tolist()
        [NaN, NaN, 3.0, 3.0, 3.0]
    """
    name = "rolling_count"
    # --> test that it works with non numeric in dfs!!!
    input_types = [ColumnSchema(logical_type=Datetime, semantic_tags={'time_index'})]
    return_type = ColumnSchema(logical_type=Double, semantic_tags={'numeric'})
    compatibility = [Library.PANDAS, Library.DASK, Library.KOALAS]

    def __init__(self, window_length, gap=0, min_periods=0):
        self.window_length = window_length
        self.gap = gap
        self.min_periods = min_periods

    def get_function(self):
        def rolling_count(datetime, numeric):
            x = pd.Series(numeric.values, index=datetime.values)
            rolled_series = roll_series_with_gap(x,
                                                 self.window_length,
                                                 gap=self.gap,
                                                 min_periods=self.min_periods)
            rolling_count_series = rolled_series.count()
            # Rolling.count will include the NaNs from the shift
            # --> account for gap=0 and min periods = 0 vs 1 vs None
            # --> get working for dask or remove from compatibility
            if not self.min_periods:
                # when min periods is 0 or None it's treated the same as if it's 1
                num_nans = self.gap
            else:
                num_nans = self.min_periods - 1 + self.gap
            rolling_count_series.iloc[range(num_nans)] = np.nan
            return rolling_count_series.values
        return rolling_count
