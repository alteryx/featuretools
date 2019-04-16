from __future__ import division

from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from ..base.aggregation_primitive_base import AggregationPrimitive

from featuretools.variable_types import (
    Boolean,
    DatetimeTimeIndex,
    Discrete,
    Index,
    Numeric,
    Variable
)


class Count(AggregationPrimitive):
    """Counts the number of values.

    Description:
        Given a list of values, return the count of values,
        excluding `NaN`.

    Examples:
        >>> count = Count()
        >>> count([1, 2, 3, 4, 5, None])
        5
    """
    name = "count"
    input_types = [[Index]]
    return_type = Numeric
    stack_on_self = False
    default_value = 0

    def get_function(self):
        return 'count'

    def generate_name(self, base_feature_names, child_entity_id,
                      parent_entity_id, where_str, use_prev_str):
        return u"COUNT(%s%s%s)" % (child_entity_id,
                                   where_str, use_prev_str)


class Sum(AggregationPrimitive):
    """Sums elements of a numeric or boolean feature.

    Description:
        Given a list of values, return the sum of values,
        ignoring `NaN`. Uses `np.sum`.

    Examples:
        >>> sum = Sum()
        >>> sum([1, 2, 3, 4, 5, None])
        15.0
    """
    name = "sum"
    input_types = [Numeric]
    return_type = Numeric
    stack_on_self = False
    stack_on_exclude = [Count]
    default_value = 0

    def get_function(self):
        return np.sum


class Mean(AggregationPrimitive):
    """Computes the average value of a numeric feature.

    Description:
        Given a list of values, return the mean of values.
        Uses `np.mean`.


    Args:
        skipna (bool): If this is False and if any value in x is `NaN`,
            return `NaN`. If this is True, `NaN` are ignored.
            Defaults to True to skip NA/null.

    Examples:
        >>> mean = Mean()
        >>> mean([1, 2, 3, 4, 5, None])
        3.0

        We can also control the way `NaN` values are handled.

        >>> mean = Mean(skipna=False)
        >>> mean([1, 2, 3, 4, 5, None])
        nan
    """
    name = "mean"
    input_types = [Numeric]
    return_type = Numeric

    def __init__(self, skipna=True):
        self.skipna = skipna

    def get_function(self):
        if self.skipna:
            # np.mean of series is functionally nanmean
            return np.mean

        def mean(series):
            return np.mean(series.values)
        return mean

    def generate_name(self, base_feature_names, child_entity_id,
                      parent_entity_id, where_str, use_prev_str):
        skipna = ""
        if not self.skipna:
            skipna = ", skipna=False"
        base_features_str = ", ".join(base_feature_names)
        return u"%s(%s.%s%s%s%s)" % (self.name.upper(),
                                     child_entity_id,
                                     base_features_str,
                                     where_str,
                                     use_prev_str,
                                     skipna)


class Mode(AggregationPrimitive):
    """Finds the mode of a feature.

    Description:
        Given a list of values, return the mode (or most
        commonly repeated value in the list). If list is
        empty, return `NaN`.

    Examples:
        >>> mode = Mode()
        >>> mode([1, 2, 3, 4, 4, 5, None])
        4.0
    """
    name = "mode"
    input_types = [Discrete]
    return_type = None

    def get_function(self):
        def pd_mode(s):
            return s.mode().get(0, np.nan)
        return pd_mode


class Min(AggregationPrimitive):
    """Finds the minimum value of a numeric feature.

    Description:
        Given a list of values, return the minimum value.
        Ignores `NaN` values.

    Examples:
        >>> min = Min()
        >>> min([1, 2, 3, 4, 5, None])
        1.0
    """
    name = "min"
    input_types = [Numeric]
    return_type = Numeric
    stack_on_self = False

    def get_function(self):
        return np.min


class Max(AggregationPrimitive):
    """Finds the maximum value of a numeric feature.

    Description:
        Given a list of values, return the maximum value.
        Ignores `NaN` values.

    Examples:
        >>> max = Max()
        >>> max([1, 2, 3, 4, 5, None])
        5.0
    """
    name = "max"
    input_types = [Numeric]
    return_type = Numeric
    stack_on_self = False

    def get_function(self):
        return np.max


class NUnique(AggregationPrimitive):
    """Finds the number of unique values.

    Description:
        Given a list of values, return the number of unique
        values. Ignores `NaN` values.

    Examples:
        >>> num_unique = NUnique()
        >>> num_unique([1, 2, 3, 4, 4, 5, None])
        5
    """
    name = "num_unique"
    input_types = [Discrete]
    return_type = Numeric
    stack_on_self = False

    def get_function(self):
        # note: returning pd.Series.nunique errors for python2,
        # so using this branching code path while we support python2
        from sys import version_info
        if version_info.major < 3:
            def nunique(x):
                return pd.Series(x).nunique()
            return nunique
        else:
            return pd.Series.nunique


class NumTrue(AggregationPrimitive):
    """Finds the number of `True` values.

    Description:
        Given a list of booleans, return the number
        of `True` values. Uses `np.sum`. Ignores 'NaN'.

    Examples:
        >>> num_true = NumTrue()
        >>> num_true([True, False, True, True, None])
        3
    """
    name = "num_true"
    input_types = [Boolean]
    return_type = Numeric
    default_value = 0
    stack_on = []
    stack_on_exclude = []

    def get_function(self):
        return np.sum


class PercentTrue(AggregationPrimitive):
    """Finds the percent of `True` values.

    Description:
        Given a list of booleans, return the percent
        of values which are `True` as a decimal. Uses
        `pd.Series.mean`. `NaN` values are treated as
        `False`, adding to the denominator.

    Examples:
        >>> percent_true = PercentTrue()
        >>> percent_true([True, False, True, True, None])
        0.6
    """
    name = "percent_true"
    input_types = [Boolean]
    return_type = Numeric
    stack_on = []
    stack_on_exclude = []
    default_value = 0

    def get_function(self):
        def percent_true(s):
            return s.fillna(0).mean()
        return percent_true


class NMostCommon(AggregationPrimitive):
    """Finds the `n` most common elements in a feature.

    Description:
        Given a list of values, return the `n` values
        which appear the most frequently. If there are
        fewer than `n` unique values, the output will be
        filled with `NaN`.

    Args:
        n (int): defines "n" in "n most common." Defaults
            to 3.

    Examples:
        >>> n_most_common = NMostCommon(n=2)
        >>> x = ['orange', 'apple', 'orange', 'apple', 'orange', 'grapefruit']
        >>> n_most_common(x).tolist()
        ['orange', 'apple']
    """
    name = "n_most_common"
    input_types = [Discrete]
    return_type = Discrete

    def __init__(self, n=3):
        self.n = n
        self.number_output_features = n

    def get_function(self):
        def n_most_common(x, n=self.number_output_features):
            array = np.array(x.value_counts()[:n].index)
            if len(array) < n:
                filler = np.full(n - len(array), np.nan)
                array = np.append(array, filler)
            return array
        return n_most_common


class AvgTimeBetween(AggregationPrimitive):
    """Computes the average time between consecutive events.

    Description:
        Given a list of datetimes, return the average time
        elapsed between consecutive events. If there are fewer
        than 2 non-null values, return `NaN`.

        Equivalent to Mean(Diff(time_index)), but more performant.

        Potentially unnecessary if we add an trans_feat that
        calculates the difference between events. DFS should
        then calculate the average of that trans_feat which
        amounts to AvgTimeBetween.

    Examples:
        >>> from datetime import datetime
        >>> avg_time_between = AvgTimeBetween()
        >>> times = [datetime(2010, 1, 1, 11, 45, 0),
        ...          datetime(2010, 1, 1, 11, 55, 15),
        ...          datetime(2010, 1, 1, 11, 57, 30)]
        >>> avg_time_between(times)
        375.0
    """
    name = "avg_time_between"
    input_types = [DatetimeTimeIndex]
    return_type = Numeric

    def get_function(self):
        def pd_avg_time_between(x):
            """Assumes time scales are closer to order
            of seconds than to nanoseconds
            if times are much closer to nanoseconds
            we could get some floating point errors

            this can be fixed with another function
            that calculates the mean before converting
            to seconds
            """
            x = x.dropna()
            if x.shape[0] < 2:
                return np.nan
            if isinstance(x.iloc[0], (pd.Timestamp, datetime)):
                x = x.astype('int64')
                # use len(x)-1 because we care about difference
                # between values, len(x)-1 = len(diff(x))

            avg = (x.max() - x.min()) / (len(x) - 1)
            avg = avg * 1e-9

            # long form:
            # diff_in_ns = x.diff().iloc[1:].astype('int64')
            # diff_in_seconds = diff_in_ns * 1e-9
            # avg = diff_in_seconds.mean()
            return avg
        return pd_avg_time_between


class Median(AggregationPrimitive):
    """Finds the median value of a feature.

    Description:
        Given a list of well-ordered values, return the median.
        Uses `pd.series.median`

    Examples:
        >>> median = Median()
        >>> median([1, 2, 3, 4, 5, None])
        3.0
    """
    name = "median"
    input_types = [Numeric]
    return_type = Numeric

    def get_function(self):
        return lambda x: x.median()


class Skew(AggregationPrimitive):
    """Computes the skewness of a feature.

    Description:
        Given a list of values, return the skew.

        For normally distributed data, the skewness should be about 0.
        A skewness value > 0 means that there is more weight in the
        left tail of the distribution.

    Examples:
        >>> skew = Skew()
        >>> skew([1, 2, 3, 4, 5, None])
        0.0
    """
    name = "skew"
    input_types = [Numeric]
    return_type = Numeric
    stack_on = []
    stack_on_self = False

    def get_function(self):
        return 'skew'


class Std(AggregationPrimitive):
    """Computes the standard deviation of a feature.

    Description:
        Given a list of values, return the standard
        deviation, ignoring `NaN`. Uses `np.std`

    Examples:
        >>> std = Std()
        >>> round(std([1, 2, 3, 4, 5, None]), 3)
        1.414
    """
    name = "std"
    input_types = [Numeric]
    return_type = Numeric
    stack_on_self = False

    def get_function(self):
        return np.std


class Last(AggregationPrimitive):
    """Returns the last value.

    Description:
        Given a list of values, return the last value.

    Examples:
        >>> last = Last()
        >>> last([1, 2, 3, 4, 5, None])
        nan
    """
    name = "last"
    input_types = [Variable]
    return_type = None
    stack_on_self = False

    def get_function(self):
        def pd_last(x):
            return x.iloc[-1]
        return pd_last


class Any(AggregationPrimitive):
    """Test if any value is 'True'.

    Description:
        Given a list of booleans, return `True` if one or
        more of the values are `True`.

    Examples:
        >>> any = Any()
        >>> any([False, False, False, True])
        True
    """
    name = "any"
    input_types = [Boolean]
    return_type = Boolean
    stack_on_self = False

    def get_function(self):
        return np.any


class All(AggregationPrimitive):
    """Test if all values are 'True'.

    Description:
        Given a list of booleans, return `True` if all
        of the values are `True`.

    Examples:
        >>> all = All()
        >>> all([False, False, False, True])
        False
    """
    name = "all"
    input_types = [Boolean]
    return_type = Boolean
    stack_on_self = False

    def get_function(self):
        return np.all


class TimeSinceLast(AggregationPrimitive):
    """Time since last instance in feature.

    Description:
        Given a list of datetimes, calculate the
        time elapsed since the last datetime (in
        seconds). Uses the instance's set cutoff time.

    Examples:
        >>> from datetime import datetime
        >>> time_since_last = TimeSinceLast()
        >>> cutoff_time = datetime(2010, 1, 1, 12, 0, 0)
        >>> times = [datetime(2010, 1, 1, 11, 45, 0),
        ...          datetime(2010, 1, 1, 11, 55, 15),
        ...          datetime(2010, 1, 1, 11, 57, 30)]
        >>> time_since_last(times, time=cutoff_time)
        150.0
    """
    name = "time_since_last"
    input_types = [DatetimeTimeIndex]
    return_type = Numeric
    uses_calc_time = True

    def get_function(self):

        def time_since_last(values, time=None):
            time_since = time - values.iloc[-1]
            return time_since.total_seconds()

        return time_since_last


class TimeSinceFirst(AggregationPrimitive):
    """Time since first instance in feature.

    Description:
        Given a list of datetimes, calculate the
        time elapsed since the first datetime (in
        seconds). Uses the instance's set cutoff time.

    Examples:
        >>> from datetime import datetime
        >>> time_since_first = TimeSinceFirst()
        >>> cutoff_time = datetime(2010, 1, 1, 12, 0, 0)
        >>> times = [datetime(2010, 1, 1, 11, 45, 0),
        ...          datetime(2010, 1, 1, 11, 55, 15),
        ...          datetime(2010, 1, 1, 11, 57, 30)]
        >>> time_since_first(times, time=cutoff_time)
        900.0
    """
    name = "time_since_first"
    input_types = [DatetimeTimeIndex]
    return_type = Numeric
    uses_calc_time = True

    def get_function(self):

        def time_since_first(values, time=None):
            time_since = time - values.iloc[0]
            return time_since.total_seconds()

        return time_since_first


class Trend(AggregationPrimitive):
    """Calculates the trend of a variable over time.

    Description:
        Given a list of values and a corresponding list of
        datetimes, calculate the slope of the linear trend
        of values.

    Examples:
        >>> from datetime import datetime
        >>> trend = Trend()
        >>> times = [datetime(2010, 1, 1, 11, 45, 0),
        ...          datetime(2010, 1, 1, 11, 55, 15),
        ...          datetime(2010, 1, 1, 11, 57, 30),
        ...          datetime(2010, 1, 1, 11, 12),
        ...          datetime(2010, 1, 1, 11, 12, 15)]
        >>> round(trend([1, 2, 3, 4, 5], times), 3)
        -0.053
    """
    name = "trend"
    input_types = [Numeric, DatetimeTimeIndex]
    return_type = Numeric

    def get_function(self):
        def pd_trend(y, x):
            df = pd.DataFrame({"x": x, "y": y}).dropna()
            if df.shape[0] <= 2:
                return np.nan
            if isinstance(df['x'].iloc[0], (datetime, pd.Timestamp)):
                x = convert_datetime_to_floats(df['x'])
            else:
                x = df['x'].values

            if isinstance(df['y'].iloc[0], (datetime, pd.Timestamp)):
                y = convert_datetime_to_floats(df['y'])
            elif isinstance(df['y'].iloc[0], (timedelta, pd.Timedelta)):
                y = convert_timedelta_to_floats(df['y'])
            else:
                y = df['y'].values

            x = x - x.mean()
            y = y - y.mean()

            # prevent divide by zero error
            if len(np.unique(x)) == 1:
                return 0

            # consider scipy.stats.linregress for large n cases
            coefficients = np.polyfit(x, y, 1)

            return coefficients[0]
        return pd_trend


def convert_datetime_to_floats(x):
    first = int(x.iloc[0].value * 1e-9)
    x = pd.to_numeric(x).astype(np.float64).values
    dividend = find_dividend_by_unit(first)
    x *= (1e-9 / dividend)
    return x


def convert_timedelta_to_floats(x):
    first = int(x.iloc[0].total_seconds())
    dividend = find_dividend_by_unit(first)
    x = pd.TimedeltaIndex(x).total_seconds().astype(np.float64) / dividend
    return x


def find_dividend_by_unit(time):
    """Finds whether time best corresponds to a value in
    days, hours, minutes, or seconds.
    """
    for dividend in [86400, 3600, 60]:
        div = time / dividend
        if round(div) == div:
            return dividend
    return 1
