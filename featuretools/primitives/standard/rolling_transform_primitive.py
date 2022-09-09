import numpy as np
import pandas as pd
from woodwork.column_schema import ColumnSchema
from woodwork.logical_types import Datetime, Double

from featuretools.primitives.base.transform_primitive_base import TransformPrimitive
from featuretools.primitives.rolling_primitive_utils import apply_rolling_agg_to_series
from featuretools.utils import calculate_trend


class RollingMax(TransformPrimitive):
    """Determines the maximum of entries over a given window.

    Description:
        Given a list of numbers and a corresponding list of
        datetimes, return a rolling maximum of the numeric values,
        starting at the row `gap` rows away from the current row and looking backward
        over the specified window (by `window_length` and `gap`).

        Input datetimes should be monotonic.

    Args:
        window_length (int, string, optional): Specifies the amount of data included in each window.
            If an integer is provided, it will correspond to a number of rows. For data with a uniform sampling frequency,
            for example of one day, the window_length will correspond to a period of time, in this case,
            7 days for a window_length of 7.
            If a string is provided, it must be one of pandas' offset alias strings ('1D', '1H', etc),
            and it will indicate a length of time that each window should span.
            The list of available offset aliases can be found at
            https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases.
            Defaults to 3.
        gap (int, string, optional): Specifies a gap backwards from each instance before the
            window of usable data begins. If an integer is provided, it will correspond to a number of rows.
            If a string is provided, it must be one of pandas' offset alias strings ('1D', '1H', etc),
            and it will indicate a length of time between a target instance and the beginning of its window.
            Defaults to 1.
        min_periods (int, optional): Minimum number of observations required for performing calculations
            over the window. Can only be as large as window_length when window_length is an integer.
            When window_length is an offset alias string, this limitation does not exist, but care should be taken
            to not choose a min_periods that will always be larger than the number of observations in a window.
            Defaults to 1.

    Note:
        Only offset aliases with fixed frequencies can be used when defining gap and window_length.
        This means that aliases such as `M` or `W` cannot be used, as they can indicate different
        numbers of days. ('M', because different months have different numbers of days;
        'W' because week will indicate a certain day of the week, like W-Wed, so that will
        indicate a different number of days depending on the anchoring date.)

    Note:
        When using an offset alias to define `gap`, an offset alias must also be used to define `window_length`.
        This limitation does not exist when using an offset alias to define `window_length`. In fact,
        if the data has a uniform sampling frequency, it is preferable to use a numeric `gap` as it is more
        efficient.

    Examples:
        >>> import pandas as pd
        >>> rolling_max = RollingMax(window_length=3)
        >>> times = pd.date_range(start='2019-01-01', freq='1min', periods=5)
        >>> rolling_max(times, [4, 3, 2, 1, 0]).tolist()
        [nan, 4.0, 4.0, 4.0, 3.0]

        We can also control the gap before the rolling calculation.

        >>> import pandas as pd
        >>> rolling_max = RollingMax(window_length=3, gap=0)
        >>> times = pd.date_range(start='2019-01-01', freq='1min', periods=5)
        >>> rolling_max(times, [4, 3, 2, 1, 0]).tolist()
        [4.0, 4.0, 4.0, 3.0, 2.0]

        We can also control the minimum number of periods required for the rolling calculation.

        >>> import pandas as pd
        >>> rolling_max = RollingMax(window_length=3, min_periods=3, gap=0)
        >>> times = pd.date_range(start='2019-01-01', freq='1min', periods=5)
        >>> rolling_max(times, [4, 3, 2, 1, 0]).tolist()
        [nan, nan, 4.0, 3.0, 2.0]

        We can also set the window_length and gap using offset alias strings.

        >>> import pandas as pd
        >>> rolling_max = RollingMax(window_length='3min', gap='1min')
        >>> times = pd.date_range(start='2019-01-01', freq='1min', periods=5)
        >>> rolling_max(times, [4, 3, 2, 1, 0]).tolist()
        [nan, 4.0, 4.0, 4.0, 3.0]
    """

    name = "rolling_max"
    input_types = [
        ColumnSchema(logical_type=Datetime, semantic_tags={"time_index"}),
        ColumnSchema(semantic_tags={"numeric"}),
    ]
    return_type = ColumnSchema(logical_type=Double, semantic_tags={"numeric"})

    def __init__(self, window_length=3, gap=1, min_periods=1):
        self.window_length = window_length
        self.gap = gap
        self.min_periods = min_periods

    def get_function(self):
        def rolling_max(datetime, numeric):
            x = pd.Series(numeric.values, index=datetime.values)
            return apply_rolling_agg_to_series(
                x,
                lambda series: series.max(),
                self.window_length,
                self.gap,
                self.min_periods,
            )

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
        window_length (int, string, optional): Specifies the amount of data included in each window.
            If an integer is provided, it will correspond to a number of rows. For data with a uniform sampling frequency,
            for example of one day, the window_length will correspond to a period of time, in this case,
            7 days for a window_length of 7.
            If a string is provided, it must be one of pandas' offset alias strings ('1D', '1H', etc),
            and it will indicate a length of time that each window should span.
            The list of available offset aliases can be found at
            https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases.
            Defaults to 3.
        gap (int, string, optional): Specifies a gap backwards from each instance before the
            window of usable data begins. If an integer is provided, it will correspond to a number of rows.
            If a string is provided, it must be one of pandas' offset alias strings ('1D', '1H', etc),
            and it will indicate a length of time between a target instance and the beginning of its window.
            Defaults to 1.
        min_periods (int, optional): Minimum number of observations required for performing calculations
            over the window. Can only be as large as window_length when window_length is an integer.
            When window_length is an offset alias string, this limitation does not exist, but care should be taken
            to not choose a min_periods that will always be larger than the number of observations in a window.
            Defaults to 1.

    Note:
        Only offset aliases with fixed frequencies can be used when defining gap and window_length.
        This means that aliases such as `M` or `W` cannot be used, as they can indicate different
        numbers of days. ('M', because different months have different numbers of days;
        'W' because week will indicate a certain day of the week, like W-Wed, so that will
        indicate a different number of days depending on the anchoring date.)

    Note:
        When using an offset alias to define `gap`, an offset alias must also be used to define `window_length`.
        This limitation does not exist when using an offset alias to define `window_length`. In fact,
        if the data has a uniform sampling frequency, it is preferable to use a numeric `gap` as it is more
        efficient.

    Examples:
        >>> import pandas as pd
        >>> rolling_min = RollingMin(window_length=3)
        >>> times = pd.date_range(start='2019-01-01', freq='1min', periods=5)
        >>> rolling_min(times, [4, 3, 2, 1, 0]).tolist()
        [nan, 4.0, 3.0, 2.0, 1.0]

        We can also control the gap before the rolling calculation.

        >>> import pandas as pd
        >>> rolling_min = RollingMin(window_length=3, gap=0)
        >>> times = pd.date_range(start='2019-01-01', freq='1min', periods=5)
        >>> rolling_min(times, [4, 3, 2, 1, 0]).tolist()
        [4.0, 3.0, 2.0, 1.0, 0.0]

        We can also control the minimum number of periods required for the rolling calculation.

        >>> import pandas as pd
        >>> rolling_min = RollingMin(window_length=3, min_periods=3, gap=0)
        >>> times = pd.date_range(start='2019-01-01', freq='1min', periods=5)
        >>> rolling_min(times, [4, 3, 2, 1, 0]).tolist()
        [nan, nan, 2.0, 1.0, 0.0]

        We can also set the window_length and gap using offset alias strings.

        >>> import pandas as pd
        >>> rolling_min = RollingMin(window_length='3min', gap='1min')
        >>> times = pd.date_range(start='2019-01-01', freq='1min', periods=5)
        >>> rolling_min(times, [4, 3, 2, 1, 0]).tolist()
        [nan, 4.0, 3.0, 2.0, 1.0]
    """

    name = "rolling_min"
    input_types = [
        ColumnSchema(logical_type=Datetime, semantic_tags={"time_index"}),
        ColumnSchema(semantic_tags={"numeric"}),
    ]
    return_type = ColumnSchema(logical_type=Double, semantic_tags={"numeric"})

    def __init__(self, window_length=3, gap=1, min_periods=1):
        self.window_length = window_length
        self.gap = gap
        self.min_periods = min_periods

    def get_function(self):
        def rolling_min(datetime, numeric):
            x = pd.Series(numeric.values, index=datetime.values)
            return apply_rolling_agg_to_series(
                x,
                lambda series: series.min(),
                self.window_length,
                self.gap,
                self.min_periods,
            )

        return rolling_min


class RollingMean(TransformPrimitive):
    """Calculates the mean of entries over a given window.

    Description:
        Given a list of numbers and a corresponding list of
        datetimes, return a rolling mean of the numeric values,
        starting at the row `gap` rows away from the current row and looking backward
        over the specified time window (by `window_length` and `gap`).

        Input datetimes should be monotonic.

    Args:
        window_length (int, string, optional): Specifies the amount of data included in each window.
            If an integer is provided, it will correspond to a number of rows. For data with a uniform sampling frequency,
            for example of one day, the window_length will correspond to a period of time, in this case,
            7 days for a window_length of 7.
            If a string is provided, it must be one of pandas' offset alias strings ('1D', '1H', etc),
            and it will indicate a length of time that each window should span.
            The list of available offset aliases can be found at
            https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases.
            Defaults to 3.
        gap (int, string, optional): Specifies a gap backwards from each instance before the
            window of usable data begins. If an integer is provided, it will correspond to a number of rows.
            If a string is provided, it must be one of pandas' offset alias strings ('1D', '1H', etc),
            and it will indicate a length of time between a target instance and the beginning of its window.
            Defaults to 1.
        min_periods (int, optional): Minimum number of observations required for performing calculations
            over the window. Can only be as large as window_length when window_length is an integer.
            When window_length is an offset alias string, this limitation does not exist, but care should be taken
            to not choose a min_periods that will always be larger than the number of observations in a window.
            Defaults to 1.

    Note:
        Only offset aliases with fixed frequencies can be used when defining gap and window_length.
        This means that aliases such as `M` or `W` cannot be used, as they can indicate different
        numbers of days. ('M', because different months have different numbers of days;
        'W' because week will indicate a certain day of the week, like W-Wed, so that will
        indicate a different number of days depending on the anchoring date.)

    Note:
        When using an offset alias to define `gap`, an offset alias must also be used to define `window_length`.
        This limitation does not exist when using an offset alias to define `window_length`. In fact,
        if the data has a uniform sampling frequency, it is preferable to use a numeric `gap` as it is more
        efficient.

    Examples:
        >>> import pandas as pd
        >>> rolling_mean = RollingMean(window_length=3)
        >>> times = pd.date_range(start='2019-01-01', freq='1min', periods=5)
        >>> rolling_mean(times, [4, 3, 2, 1, 0]).tolist()
        [nan, 4.0, 3.5, 3.0, 2.0]

        We can also control the gap before the rolling calculation.

        >>> import pandas as pd
        >>> rolling_mean = RollingMean(window_length=3, gap=0)
        >>> times = pd.date_range(start='2019-01-01', freq='1min', periods=5)
        >>> rolling_mean(times, [4, 3, 2, 1, 0]).tolist()
        [4.0, 3.5, 3.0, 2.0, 1.0]

        We can also control the minimum number of periods required for the rolling calculation.

        >>> import pandas as pd
        >>> rolling_mean = RollingMean(window_length=3, min_periods=3, gap=0)
        >>> times = pd.date_range(start='2019-01-01', freq='1min', periods=5)
        >>> rolling_mean(times, [4, 3, 2, 1, 0]).tolist()
        [nan, nan, 3.0, 2.0, 1.0]
    """

    name = "rolling_mean"
    input_types = [
        ColumnSchema(logical_type=Datetime, semantic_tags={"time_index"}),
        ColumnSchema(semantic_tags={"numeric"}),
    ]
    return_type = ColumnSchema(logical_type=Double, semantic_tags={"numeric"})

    def __init__(self, window_length=3, gap=1, min_periods=0):
        self.window_length = window_length
        self.gap = gap
        self.min_periods = min_periods

    def get_function(self):
        def rolling_mean(datetime, numeric):
            x = pd.Series(numeric.values, index=datetime.values)
            return apply_rolling_agg_to_series(
                x,
                np.mean,
                self.window_length,
                self.gap,
                self.min_periods,
            )

        return rolling_mean


class RollingSTD(TransformPrimitive):
    """Calculates the standard deviation of entries over a given window.

    Description:
        Given a list of numbers and a corresponding list of
        datetimes, return a rolling standard deviation of
        the numeric values, starting at the row `gap` rows away from the current row and
        looking backward over the specified time window
        (by `window_length` and `gap`). Input datetimes should be monotonic.

    Args:
        window_length (int, string, optional): Specifies the amount of data included in each window.
            If an integer is provided, it will correspond to a number of rows. For data with a uniform sampling frequency,
            for example of one day, the window_length will correspond to a period of time, in this case,
            7 days for a window_length of 7.
            If a string is provided, it must be one of pandas' offset alias strings ('1D', '1H', etc),
            and it will indicate a length of time that each window should span.
            The list of available offset aliases can be found at
            https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases.
            Defaults to 3.
        gap (int, string, optional): Specifies a gap backwards from each instance before the
            window of usable data begins. If an integer is provided, it will correspond to a number of rows.
            If a string is provided, it must be one of pandas' offset alias strings ('1D', '1H', etc),
            and it will indicate a length of time between a target instance and the beginning of its window.
            Defaults to 1.
        min_periods (int, optional): Minimum number of observations required for performing calculations
            over the window. Can only be as large as window_length when window_length is an integer.
            When window_length is an offset alias string, this limitation does not exist, but care should be taken
            to not choose a min_periods that will always be larger than the number of observations in a window.
            Defaults to 1.

    Note:
        Only offset aliases with fixed frequencies can be used when defining gap and window_length.
        This means that aliases such as `M` or `W` cannot be used, as they can indicate different
        numbers of days. ('M', because different months have different numbers of days;
        'W' because week will indicate a certain day of the week, like W-Wed, so that will
        indicate a different number of days depending on the anchoring date.)

    Note:
        When using an offset alias to define `gap`, an offset alias must also be used to define `window_length`.
        This limitation does not exist when using an offset alias to define `window_length`. In fact,
        if the data has a uniform sampling frequency, it is preferable to use a numeric `gap` as it is more
        efficient.

    Examples:
        >>> import pandas as pd
        >>> rolling_std = RollingSTD(window_length=4)
        >>> times = pd.date_range(start='2019-01-01', freq='1min', periods=5)
        >>> rolling_std(times, [4, 3, 2, 1, 0]).tolist()
        [nan, nan, 0.7071067811865476, 1.0, 1.2909944487358056]

        We can also control the gap before the rolling calculation.

        >>> import pandas as pd
        >>> rolling_std = RollingSTD(window_length=4, gap=0)
        >>> times = pd.date_range(start='2019-01-01', freq='1min', periods=5)
        >>> rolling_std(times, [4, 3, 2, 1, 0]).tolist()
        [nan, 0.7071067811865476, 1.0, 1.2909944487358056, 1.2909944487358056]

        We can also control the minimum number of periods required for the rolling calculation.

        >>> import pandas as pd
        >>> rolling_std = RollingSTD(window_length=4, min_periods=4, gap=0)
        >>> times = pd.date_range(start='2019-01-01', freq='1min', periods=5)
        >>> rolling_std(times, [4, 3, 2, 1, 0]).tolist()
        [nan, nan, nan, 1.2909944487358056, 1.2909944487358056]

        We can also set the window_length and gap using offset alias strings.
        >>> import pandas as pd
        >>> rolling_std = RollingSTD(window_length='4min', gap='1min')
        >>> times = pd.date_range(start='2019-01-01', freq='1min', periods=5)
        >>> rolling_std(times, [4, 3, 2, 1, 0]).tolist()
        [nan, nan, 0.7071067811865476, 1.0, 1.2909944487358056]
    """

    name = "rolling_std"
    input_types = [
        ColumnSchema(logical_type=Datetime, semantic_tags={"time_index"}),
        ColumnSchema(semantic_tags={"numeric"}),
    ]
    return_type = ColumnSchema(logical_type=Double, semantic_tags={"numeric"})

    def __init__(self, window_length=3, gap=1, min_periods=1):
        self.window_length = window_length
        self.gap = gap
        self.min_periods = min_periods

    def get_function(self):
        def rolling_std(datetime, numeric):
            x = pd.Series(numeric.values, index=datetime.values)
            return apply_rolling_agg_to_series(
                x,
                lambda series: series.std(),
                self.window_length,
                self.gap,
                self.min_periods,
            )

        return rolling_std


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
            If an integer is provided, it will correspond to a number of rows. For data with a uniform sampling frequency,
            for example of one day, the window_length will correspond to a period of time, in this case,
            7 days for a window_length of 7.
            If a string is provided, it must be one of pandas' offset alias strings ('1D', '1H', etc),
            and it will indicate a length of time that each window should span.
            The list of available offset aliases can be found at
            https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases.
            Defaults to 3.
        gap (int, string, optional): Specifies a gap backwards from each instance before the
            window of usable data begins. If an integer is provided, it will correspond to a number of rows.
            If a string is provided, it must be one of pandas' offset alias strings ('1D', '1H', etc),
            and it will indicate a length of time between a target instance and the beginning of its window.
            Defaults to 1.
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
        [nan, nan, nan, 1.4999999999999998, 2.9999999999999996, 5.999999999999999, 7.999999999999999, 16.0, 36.0, 72.0]

        We can also control the gap before the rolling calculation.

        >>> rolling_trend = RollingTrend(gap=0)
        >>> rolling_trend(times, [1, 2, 4, 8, 16, 24, 48, 96, 192, 384]).tolist()
        [nan, nan, 1.4999999999999998, 2.9999999999999996, 5.999999999999999, 7.999999999999999, 16.0, 36.0, 72.0, 144.0]

        We can also control the minimum number of periods required for the rolling calculation.

        >>> rolling_trend = RollingTrend(window_length=4, min_periods=4, gap=0)
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

    def __init__(self, window_length=3, gap=1, min_periods=0):
        self.window_length = window_length
        self.gap = gap
        self.min_periods = min_periods

    def get_function(self):
        def rolling_trend(datetime, numeric):
            x = pd.Series(numeric.values, index=datetime.values)
            return apply_rolling_agg_to_series(
                x,
                calculate_trend,
                self.window_length,
                self.gap,
                self.min_periods,
            )

        return rolling_trend


class RollingCount(TransformPrimitive):
    """Determines a rolling count of events over a given window.

    Description:
        Given a list of datetimes, return a rolling count starting
        at the row `gap` rows away from the current row and looking backward over the specified
        time window (by `window_length` and `gap`).

        Input datetimes should be monotonic.

    Args:
        window_length (int, string, optional): Specifies the amount of data included in each window.
            If an integer is provided, it will correspond to a number of rows. For data with a uniform sampling frequency,
            for example of one day, the window_length will correspond to a period of time, in this case,
            7 days for a window_length of 7.
            If a string is provided, it must be one of pandas' offset alias strings ('1D', '1H', etc),
            and it will indicate a length of time that each window should span.
            The list of available offset aliases can be found at
            https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases.
            Defaults to 3.
        gap (int, string, optional): Specifies a gap backwards from each instance before the
            window of usable data begins. If an integer is provided, it will correspond to a number of rows.
            If a string is provided, it must be one of pandas' offset alias strings ('1D', '1H', etc),
            and it will indicate a length of time between a target instance and the beginning of its window.
            Defaults to 1.
        min_periods (int, optional): Minimum number of observations required for performing calculations
            over the window. Can only be as large as window_length when window_length is an integer.
            When window_length is an offset alias string, this limitation does not exist, but care should be taken
            to not choose a min_periods that will always be larger than the number of observations in a window.
            Defaults to 1.

    Note:
        Only offset aliases with fixed frequencies can be used when defining gap and h.
        This means that aliases such as `M` or `W` cannot be used, as they can indicate different
        numbers of days. ('M', because different months have different numbers of days;
        'W' because week will indicate a certain day of the week, like W-Wed, so that will
        indicate a different number of days depending on the anchoring date.)

    Note:
        When using an offset alias to define `gap`, an offset alias must also be used to define `window_length`.
        This limitation does not exist when using an offset alias to define `window_length`. In fact,
        if the data has a uniform sampling frequency, it is preferable to use a numeric `gap` as it is more
        efficient.

    Examples:
        >>> import pandas as pd
        >>> rolling_count = RollingCount(window_length=3)
        >>> times = pd.date_range(start='2019-01-01', freq='1min', periods=5)
        >>> rolling_count(times).tolist()
        [nan, 1.0, 2.0, 3.0, 3.0]

        We can also control the gap before the rolling calculation.

        >>> import pandas as pd
        >>> rolling_count = RollingCount(window_length=3, gap=0)
        >>> times = pd.date_range(start='2019-01-01', freq='1min', periods=5)
        >>> rolling_count(times).tolist()
        [1.0, 2.0, 3.0, 3.0, 3.0]

        We can also control the minimum number of periods required for the rolling calculation.

        >>> import pandas as pd
        >>> rolling_count = RollingCount(window_length=3, min_periods=3, gap=0)
        >>> times = pd.date_range(start='2019-01-01', freq='1min', periods=5)
        >>> rolling_count(times).tolist()
        [nan, nan, 3.0, 3.0, 3.0]

        We can also set the window_length and gap using offset alias strings.
        >>> import pandas as pd
        >>> rolling_count = RollingCount(window_length='3min', gap='1min')
        >>> times = pd.date_range(start='2019-01-01', freq='1min', periods=5)
        >>> rolling_count(times).tolist()
        [nan, 1.0, 2.0, 3.0, 3.0]

    """

    name = "rolling_count"
    input_types = [ColumnSchema(logical_type=Datetime, semantic_tags={"time_index"})]
    return_type = ColumnSchema(logical_type=Double, semantic_tags={"numeric"})

    def __init__(self, window_length=3, gap=1, min_periods=0):
        self.window_length = window_length
        self.gap = gap
        self.min_periods = min_periods

    def get_function(self):
        def rolling_count(datetime):
            x = pd.Series(1, index=datetime)
            return apply_rolling_agg_to_series(
                x,
                lambda series: series.count(),
                self.window_length,
                self.gap,
                self.min_periods,
                ignore_window_nans=True,
            )

        return rolling_count
