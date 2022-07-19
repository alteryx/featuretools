import numpy as np
from pandas import Series
from pandas.tseries.frequencies import to_offset


def roll_series_with_gap(series, window_size, gap=0, min_periods=1):
    """Provide rolling window calculations where the windows are determined using both a gap parameter
    that indicates the amount of time between each instance and its window and a window length parameter
    that determines the amount of data in each window.

    Args:
        series (Series): The series over which rolling windows will be created. Must be numeric in nature
            and have a DatetimeIndex.
        window_size (int, string): Specifies the amount of data included in each window.
            If an integer is provided, will correspond to a number of rows. For data with a uniform sampling frequency,
            for example of one day, the window_length will correspond to a period of time, in this case,
            7 days for a window_length of 7.
            If a string is provided, it must be one of pandas' offset alias strings ('1D', '1H', etc),
            and it will indicate a length of time that each window should span.
            The list of available offset aliases, can be found at
            https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases
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

    Returns:
        pandas.core.window.rolling.Rolling: The Rolling object for the series passed in.

    Note:
        Certain operations, like `pandas.core.window.rolling.Rolling.count` that can be performed
        on the Rolling object returned here may treat NaNs as periods to include in window calculations.
        So a window [NaN, 1, 3]  when `min_periods=3` will proceed with count, saying there are three periods
        but only two values and would return count=2. The calculation `max` on the other hand,
        would say that there are not three periods in that window and would return max=NaN.
        Most rolling calculations act this way. The implication of that here is that in order to
        achieve the gap, we insert NaNs at the beinning of the series, which would cause `count` to calculate
        on windows that technically should not have the correct number of periods. In the RollingCount primitive,
        we handle this case manually, replacing those values with NaNs. Any primitive that uses this function
        should determine whether this kind of handling is also necessary.

    Note:
        Only offset aliases with fixed frequencies can be used when defining gap and window_length.
        This means that aliases such as `M` or `W` cannot be used, as they can indicate different
        numbers of days. ('M', because different months are different numbers of days;
        'W' because week will indicate a certain day of the week, like W-Wed, so that will
        indicate a different number of days depending on the anchoring date.)

    Note:
        When using an offset alias to define `gap`, an offset alias must also be used to define `window_size`.
        This limitation does not exist when using an offset alias to define `window_size`. In fact,
        if the data has a uniform sampling frequency, it is preferable to use a numeric `gap` as it is more
        efficient.

    """
    _check_window_size(window_size)
    _check_gap(window_size, gap)

    # Workaround for pandas' bug: https://github.com/pandas-dev/pandas/issues/43016
    # Can remove when upgraded to pandas 1.4.0
    if str(series.dtype) == "Int64":
        series = series.astype("float64")

    functional_window_length = window_size
    if isinstance(gap, str):
        # Add the window_size and gap so that the rolling operation correctly takes gap into account.
        # That way, we can later remove the gap rows in order to apply the primitive function
        # to the correct window
        functional_window_length = to_offset(window_size) + to_offset(gap)
    elif gap > 0:
        # When gap is numeric, we can apply a shift to incorporate gap right now
        # since the gap will be the same number of rows for the whole dataset
        series = series.shift(gap)

    return series.rolling(functional_window_length, min_periods)


def _get_rolled_series_without_gap(window, gap_offset):
    """Applies the gap offset_string to the rolled window, returning a window
    that is the correct length of time away from the original instance.

    Args:
        window (Series): A rolling window that includes both the window length and gap spans of time.
        gap_offset (string): The pandas offset alias that determines how much time at the end of the window
            should be removed.

    Returns:
        Series: The window with gap rows removed
    """
    if not len(window):
        return window

    window_start_date = window.index[0]
    window_end_date = window.index[-1]

    gap_bound = window_end_date - to_offset(gap_offset)

    # If the gap is larger than the series, no rows are left in the window
    if gap_bound < window_start_date:
        return Series(dtype="float64")

    # Only return the rows that are within the offset's bounds
    return window[window.index <= gap_bound]


def apply_roll_with_offset_gap(window, gap_offset, reducer_fn, min_periods):
    """Takes in a series to which an offset gap will be applied, removing however many
    rows fall under the gap before applying the reducing function.

    Args:
        window (Series):  A rolling window that includes both the window length and gap spans of time.
        gap_offset (string): The pandas offset alias that determines how much time at the end of the window
            should be removed.
        reducer_fn (callable[Series -> float]): The function to be applied to the window in order to produce
            the aggregate that will be included in the resulting feature.
        min_periods (int): Minimum number of observations required for performing calculations
            over the window.

    Returns:
        float: The aggregate value to be used as a feature value.
    """
    window = _get_rolled_series_without_gap(window, gap_offset)

    if min_periods is None:
        min_periods = 1

    if len(window) < min_periods or not len(window):
        return np.nan

    return reducer_fn(window)


def _check_window_size(window_size):
    # Window length must either be a valid offset alias
    if isinstance(window_size, str):
        try:
            to_offset(window_size)
        except ValueError:
            raise ValueError(
                f"Cannot roll series. The specified window length, {window_size}, is not a valid offset alias."
            )
    # Or an integer greater than zero
    elif isinstance(window_size, int):
        if window_size <= 0:
            raise ValueError("Window length must be greater than zero.")
    else:
        raise TypeError("Window length must be either an offset string or an integer.")


def _check_gap(window_size, gap):
    # Gap must either be a valid offset string that also has an offset string window length
    if isinstance(gap, str):
        if not isinstance(window_size, str):
            raise TypeError(
                f"Cannot roll series with offset gap, {gap}, and numeric window length, {window_size}. "
                "If an offset alias is used for gap, the window length must also be defined as an offset alias. "
                "Please either change gap to be numeric or change window length to be an offset alias."
            )
        try:
            to_offset(gap)
        except ValueError:
            raise ValueError(
                f"Cannot roll series. The specified gap, {gap}, is not a valid offset alias."
            )
    # Or an integer greater than or equal to zero
    elif isinstance(gap, int):
        if gap < 0:
            raise ValueError("Gap must be greater than or equal to zero.")
    else:
        raise TypeError("Gap must be either an offset string or an integer.")
