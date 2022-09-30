from datetime import datetime

import numpy as np
import pandas as pd
from dask import dataframe as dd
from scipy import stats
from woodwork.column_schema import ColumnSchema
from woodwork.logical_types import (
    Boolean,
    BooleanNullable,
    Datetime,
    Double,
    IntegerNullable,
)

from featuretools.primitives.base.aggregation_primitive_base import AggregationPrimitive
from featuretools.utils import calculate_trend, convert_time_units
from featuretools.utils.gen_utils import Library


class Count(AggregationPrimitive):
    """Determines the total number of values, excluding `NaN`.

    Examples:
        >>> count = Count()
        >>> count([1, 2, 3, 4, 5, None])
        5
    """

    name = "count"
    input_types = [ColumnSchema(semantic_tags={"index"})]
    return_type = ColumnSchema(logical_type=IntegerNullable, semantic_tags={"numeric"})
    stack_on_self = False
    default_value = 0
    compatibility = [Library.PANDAS, Library.DASK, Library.SPARK]
    description_template = "the number"

    def get_function(self, agg_type=Library.PANDAS):
        if agg_type in [Library.DASK, Library.SPARK]:
            return "count"

        return pd.Series.count

    def generate_name(
        self,
        base_feature_names,
        relationship_path_name,
        parent_dataframe_name,
        where_str,
        use_prev_str,
    ):
        return "COUNT(%s%s%s)" % (relationship_path_name, where_str, use_prev_str)


class Sum(AggregationPrimitive):
    """Calculates the total addition, ignoring `NaN`.

    Examples:
        >>> sum = Sum()
        >>> sum([1, 2, 3, 4, 5, None])
        15.0
    """

    name = "sum"
    input_types = [ColumnSchema(semantic_tags={"numeric"})]
    return_type = ColumnSchema(semantic_tags={"numeric"})
    stack_on_self = False
    stack_on_exclude = [Count]
    default_value = 0
    compatibility = [Library.PANDAS, Library.DASK, Library.SPARK]
    description_template = "the sum of {}"

    def get_function(self, agg_type=Library.PANDAS):
        if agg_type in [Library.DASK, Library.SPARK]:
            return "sum"

        return np.sum


class Mean(AggregationPrimitive):
    """Computes the average for a list of values.

    Args:
        skipna (bool): Determines if to use NA/null values. Defaults to
            True to skip NA/null.

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
    input_types = [ColumnSchema(semantic_tags={"numeric"})]
    return_type = ColumnSchema(semantic_tags={"numeric"})
    compatibility = [Library.PANDAS, Library.DASK, Library.SPARK]
    description_template = "the average of {}"

    def __init__(self, skipna=True):
        self.skipna = skipna

    def get_function(self, agg_type=Library.PANDAS):
        if agg_type in [Library.DASK, Library.SPARK]:
            return "mean"

        if self.skipna:
            # np.mean of series is functionally nanmean
            return np.mean

        def mean(series):
            return np.mean(series.values)

        return mean


class Mode(AggregationPrimitive):
    """Determines the most commonly repeated value.

    Description:
        Given a list of values, return the value with the
        highest number of occurences. If list is
        empty, return `NaN`.

    Examples:
        >>> mode = Mode()
        >>> mode(['red', 'blue', 'green', 'blue'])
        'blue'
    """

    name = "mode"
    input_types = [ColumnSchema(semantic_tags={"category"})]
    return_type = None
    description_template = "the most frequently occurring value of {}"

    def get_function(self, agg_type=Library.PANDAS):
        def pd_mode(s):
            return s.mode().get(0, np.nan)

        return pd_mode


class Min(AggregationPrimitive):
    """Calculates the smallest value, ignoring `NaN` values.

    Examples:
        >>> min = Min()
        >>> min([1, 2, 3, 4, 5, None])
        1.0
    """

    name = "min"
    input_types = [ColumnSchema(semantic_tags={"numeric"})]
    return_type = ColumnSchema(semantic_tags={"numeric"})
    stack_on_self = False
    compatibility = [Library.PANDAS, Library.DASK, Library.SPARK]
    description_template = "the minimum of {}"

    def get_function(self, agg_type=Library.PANDAS):
        if agg_type in [Library.DASK, Library.SPARK]:
            return "min"

        return np.min


class Max(AggregationPrimitive):
    """Calculates the highest value, ignoring `NaN` values.

    Examples:
        >>> max = Max()
        >>> max([1, 2, 3, 4, 5, None])
        5.0
    """

    name = "max"
    input_types = [ColumnSchema(semantic_tags={"numeric"})]
    return_type = ColumnSchema(semantic_tags={"numeric"})
    stack_on_self = False
    compatibility = [Library.PANDAS, Library.DASK, Library.SPARK]
    description_template = "the maximum of {}"

    def get_function(self, agg_type=Library.PANDAS):
        if agg_type in [Library.DASK, Library.SPARK]:
            return "max"

        return np.max


class NumUnique(AggregationPrimitive):
    """Determines the number of distinct values, ignoring `NaN` values.

    Examples:
        >>> num_unique = NumUnique()
        >>> num_unique(['red', 'blue', 'green', 'yellow'])
        4

        `NaN` values will be ignored.

        >>> num_unique(['red', 'blue', 'green', 'yellow', None])
        4
    """

    name = "num_unique"
    input_types = [ColumnSchema(semantic_tags={"category"})]
    return_type = ColumnSchema(logical_type=IntegerNullable, semantic_tags={"numeric"})
    stack_on_self = False
    compatibility = [Library.PANDAS, Library.DASK, Library.SPARK]
    description_template = "the number of unique elements in {}"

    def get_function(self, agg_type=Library.PANDAS):
        if agg_type == Library.DASK:

            def chunk(s):
                def inner_chunk(x):
                    x = x[:].dropna()
                    return set(x.unique())

                return s.agg(inner_chunk)

            def agg(s):
                def inner_agg(x):
                    x = x[:].dropna()
                    return set().union(*x.values)

                return s.agg(inner_agg)

            def finalize(s):
                return s.apply(lambda x: len(x))

            return dd.Aggregation(self.name, chunk=chunk, agg=agg, finalize=finalize)

        elif agg_type == Library.SPARK:
            return "nunique"

        return pd.Series.nunique


class NumTrue(AggregationPrimitive):
    """Counts the number of `True` values.

    Description:
        Given a list of booleans, return the number
        of `True` values. Ignores 'NaN'.

    Examples:
        >>> num_true = NumTrue()
        >>> num_true([True, False, True, True, None])
        3
    """

    name = "num_true"
    input_types = [
        [ColumnSchema(logical_type=Boolean)],
        [ColumnSchema(logical_type=BooleanNullable)],
    ]
    return_type = ColumnSchema(logical_type=IntegerNullable, semantic_tags={"numeric"})
    default_value = 0
    stack_on = []
    stack_on_exclude = []
    compatibility = [Library.PANDAS, Library.DASK]
    description_template = "the number of times {} is true"

    def get_function(self, agg_type=Library.PANDAS):
        if agg_type == Library.DASK:

            def chunk(s):
                chunk_sum = s.agg(np.sum)
                if chunk_sum.dtype == "bool":
                    chunk_sum = chunk_sum.astype("int64")
                return chunk_sum

            def agg(s):
                return s.agg(np.sum)

            return dd.Aggregation(self.name, chunk=chunk, agg=agg)

        return np.sum


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
    default_value = 0
    compatibility = [Library.PANDAS, Library.DASK]
    description_template = "the percentage of true values in {}"

    def get_function(self, agg_type=Library.PANDAS):
        if agg_type == Library.DASK:

            def chunk(s):
                def format_chunk(x):
                    return x[:].fillna(False)

                chunk_sum = s.agg(lambda x: format_chunk(x).sum())
                chunk_len = s.agg(lambda x: len(format_chunk(x)))
                if chunk_sum.dtype == "bool":
                    chunk_sum = chunk_sum.astype("int64")
                if chunk_len.dtype == "bool":
                    chunk_len = chunk_len.astype("int64")
                return (chunk_sum, chunk_len)

            def agg(val, length):
                return (val.sum(), length.sum())

            def finalize(total, length):
                return total / length

            return dd.Aggregation(self.name, chunk=chunk, agg=agg, finalize=finalize)

        def percent_true(s):
            return s.fillna(False).mean()

        return percent_true


class NMostCommon(AggregationPrimitive):
    """Determines the `n` most common elements.

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
    input_types = [ColumnSchema(semantic_tags={"category"})]
    return_type = None

    def __init__(self, n=3):
        self.n = n
        self.number_output_features = n
        self.description_template = [
            "the {} most common values of {{}}".format(n),
            "the most common value of {}",
            *["the {nth_slice} most common value of {}"] * (n - 1),
        ]

    def get_function(self, agg_type=Library.PANDAS):
        def n_most_common(x):
            # Counts of 0 remain in value_counts output if dtype is category
            # so we need to remove them
            counts = x.value_counts()
            counts = counts[counts > 0]
            array = np.array(counts.index[: self.n])
            if len(array) < self.n:
                filler = np.full(self.n - len(array), np.nan)
                array = np.append(array, filler)
            return array

        return n_most_common


class AvgTimeBetween(AggregationPrimitive):
    """Computes the average number of seconds between consecutive events.

    Description:
        Given a list of datetimes, return the average time (default in seconds)
        elapsed between consecutive events. If there are fewer
        than 2 non-null values, return `NaN`.

    Args:
        unit (str): Defines the unit of time.
            Defaults to seconds. Acceptable values:
            years, months, days, hours, minutes, seconds, milliseconds, nanoseconds

    Examples:
        >>> from datetime import datetime
        >>> avg_time_between = AvgTimeBetween()
        >>> times = [datetime(2010, 1, 1, 11, 45, 0),
        ...          datetime(2010, 1, 1, 11, 55, 15),
        ...          datetime(2010, 1, 1, 11, 57, 30)]
        >>> avg_time_between(times)
        375.0
        >>> avg_time_between = AvgTimeBetween(unit="minutes")
        >>> avg_time_between(times)
        6.25
    """

    name = "avg_time_between"
    input_types = [ColumnSchema(logical_type=Datetime, semantic_tags={"time_index"})]
    return_type = ColumnSchema(logical_type=Double, semantic_tags={"numeric"})
    description_template = "the average time between each of {}"

    def __init__(self, unit="seconds"):
        self.unit = unit.lower()

    def get_function(self, agg_type=Library.PANDAS):
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
                x = x.view("int64")
                # use len(x)-1 because we care about difference
                # between values, len(x)-1 = len(diff(x))

            avg = (x.max() - x.min()) / (len(x) - 1)
            avg = avg * 1e-9

            # long form:
            # diff_in_ns = x.diff().iloc[1:].astype('int64')
            # diff_in_seconds = diff_in_ns * 1e-9
            # avg = diff_in_seconds.mean()
            return convert_time_units(avg, self.unit)

        return pd_avg_time_between


class Median(AggregationPrimitive):
    """Determines the middlemost number in a list of values.

    Examples:
        >>> median = Median()
        >>> median([5, 3, 2, 1, 4])
        3.0

        `NaN` values are ignored.

        >>> median([5, 3, 2, 1, 4, None])
        3.0
    """

    name = "median"
    input_types = [ColumnSchema(semantic_tags={"numeric"})]
    return_type = ColumnSchema(semantic_tags={"numeric"})
    description_template = "the median of {}"

    def get_function(self, agg_type=Library.PANDAS):
        return pd.Series.median


class Skew(AggregationPrimitive):
    """Computes the extent to which a distribution differs from a normal distribution.

    Description:
        For normally distributed data, the skewness should be about 0.
        A skewness value > 0 means that there is more weight in the
        left tail of the distribution.

    Examples:
        >>> skew = Skew()
        >>> skew([1, 10, 30, None])
        1.0437603722639681
    """

    name = "skew"
    input_types = [ColumnSchema(semantic_tags={"numeric"})]
    return_type = ColumnSchema(semantic_tags={"numeric"})
    stack_on = []
    stack_on_self = False
    description_template = "the skewness of {}"

    def get_function(self, agg_type=Library.PANDAS):
        return pd.Series.skew


class Std(AggregationPrimitive):
    """Computes the dispersion relative to the mean value, ignoring `NaN`.

    Examples:
        >>> std = Std()
        >>> round(std([1, 2, 3, 4, 5, None]), 3)
        1.414
    """

    name = "std"
    input_types = [ColumnSchema(semantic_tags={"numeric"})]
    return_type = ColumnSchema(semantic_tags={"numeric"})
    stack_on_self = False
    compatibility = [Library.PANDAS, Library.DASK, Library.SPARK]
    description_template = "the standard deviation of {}"

    def get_function(self, agg_type=Library.PANDAS):
        if agg_type in [Library.DASK, Library.SPARK]:
            return "std"

        return np.std


class First(AggregationPrimitive):
    """Determines the first value in a list.

    Examples:
        >>> first = First()
        >>> first([1, 2, 3, 4, 5, None])
        1.0
    """

    name = "first"
    input_types = [ColumnSchema()]
    return_type = None
    stack_on_self = False
    description_template = "the first instance of {}"

    def get_function(self, agg_type=Library.PANDAS):
        def pd_first(x):
            return x.iloc[0]

        return pd_first


class Last(AggregationPrimitive):
    """Determines the last value in a list.

    Examples:
        >>> last = Last()
        >>> last([1, 2, 3, 4, 5, None])
        nan
    """

    name = "last"
    input_types = [ColumnSchema()]
    return_type = None
    stack_on_self = False
    description_template = "the last instance of {}"

    def get_function(self, agg_type=Library.PANDAS):
        def pd_last(x):
            return x.iloc[-1]

        return pd_last


class Any(AggregationPrimitive):
    """Determines if any value is 'True' in a list.

    Description:
        Given a list of booleans, return `True` if one or
        more of the values are `True`.

    Examples:
        >>> any = Any()
        >>> any([False, False, False, True])
        True
    """

    name = "any"
    input_types = [
        [ColumnSchema(logical_type=Boolean)],
        [ColumnSchema(logical_type=BooleanNullable)],
    ]
    return_type = ColumnSchema(logical_type=Boolean)
    stack_on_self = False
    compatibility = [Library.PANDAS, Library.DASK]
    description_template = "whether any of {} are true"

    def get_function(self, agg_type=Library.PANDAS):
        if agg_type == Library.DASK:

            def chunk(s):
                return s.agg(np.any)

            def agg(s):
                return s.agg(np.any)

            return dd.Aggregation(self.name, chunk=chunk, agg=agg)

        return np.any


class All(AggregationPrimitive):
    """Calculates if all values are 'True' in a list.

    Description:
        Given a list of booleans, return `True` if all
        of the values are `True`.

    Examples:
        >>> all = All()
        >>> all([False, False, False, True])
        False
    """

    name = "all"
    input_types = [
        [ColumnSchema(logical_type=Boolean)],
        [ColumnSchema(logical_type=BooleanNullable)],
    ]
    return_type = ColumnSchema(logical_type=Boolean)
    stack_on_self = False
    compatibility = [Library.PANDAS, Library.DASK]
    description_template = "whether all of {} are true"

    def get_function(self, agg_type=Library.PANDAS):
        if agg_type == Library.DASK:

            def chunk(s):
                return s.agg(np.all)

            def agg(s):
                return s.agg(np.all)

            return dd.Aggregation(self.name, chunk=chunk, agg=agg)

        return np.all


class TimeSinceLast(AggregationPrimitive):
    """Calculates the time elapsed since the last datetime (default in seconds).

    Description:
        Given a list of datetimes, calculate the
        time elapsed since the last datetime (default in
        seconds). Uses the instance's cutoff time.

    Args:
        unit (str): Defines the unit of time to count from.
            Defaults to seconds. Acceptable values:
            years, months, days, hours, minutes, seconds, milliseconds, nanoseconds

    Examples:
        >>> from datetime import datetime
        >>> time_since_last = TimeSinceLast()
        >>> cutoff_time = datetime(2010, 1, 1, 12, 0, 0)
        >>> times = [datetime(2010, 1, 1, 11, 45, 0),
        ...          datetime(2010, 1, 1, 11, 55, 15),
        ...          datetime(2010, 1, 1, 11, 57, 30)]
        >>> time_since_last(times, time=cutoff_time)
        150.0

        >>> from datetime import datetime
        >>> time_since_last = TimeSinceLast(unit = "minutes")
        >>> cutoff_time = datetime(2010, 1, 1, 12, 0, 0)
        >>> times = [datetime(2010, 1, 1, 11, 45, 0),
        ...          datetime(2010, 1, 1, 11, 55, 15),
        ...          datetime(2010, 1, 1, 11, 57, 30)]
        >>> time_since_last(times, time=cutoff_time)
        2.5

    """

    name = "time_since_last"
    input_types = [ColumnSchema(logical_type=Datetime, semantic_tags={"time_index"})]
    return_type = ColumnSchema(logical_type=Double, semantic_tags={"numeric"})
    uses_calc_time = True
    description_template = "the time since the last {}"

    def __init__(self, unit="seconds"):
        self.unit = unit.lower()

    def get_function(self, agg_type=Library.PANDAS):
        def time_since_last(values, time=None):
            time_since = time - values.iloc[-1]
            return convert_time_units(time_since.total_seconds(), self.unit)

        return time_since_last


class TimeSinceFirst(AggregationPrimitive):
    """Calculates the time elapsed since the first datetime (in seconds).

    Description:
        Given a list of datetimes, calculate the
        time elapsed since the first datetime (in
        seconds). Uses the instance's cutoff time.

    Args:
        unit (str): Defines the unit of time to count from.
            Defaults to seconds. Acceptable values:
            years, months, days, hours, minutes, seconds, milliseconds, nanoseconds

    Examples:
        >>> from datetime import datetime
        >>> time_since_first = TimeSinceFirst()
        >>> cutoff_time = datetime(2010, 1, 1, 12, 0, 0)
        >>> times = [datetime(2010, 1, 1, 11, 45, 0),
        ...          datetime(2010, 1, 1, 11, 55, 15),
        ...          datetime(2010, 1, 1, 11, 57, 30)]
        >>> time_since_first(times, time=cutoff_time)
        900.0

        >>> from datetime import datetime
        >>> time_since_first = TimeSinceFirst(unit = "minutes")
        >>> cutoff_time = datetime(2010, 1, 1, 12, 0, 0)
        >>> times = [datetime(2010, 1, 1, 11, 45, 0),
        ...          datetime(2010, 1, 1, 11, 55, 15),
        ...          datetime(2010, 1, 1, 11, 57, 30)]
        >>> time_since_first(times, time=cutoff_time)
        15.0

    """

    name = "time_since_first"
    input_types = [ColumnSchema(logical_type=Datetime, semantic_tags={"time_index"})]
    return_type = ColumnSchema(logical_type=Double, semantic_tags={"numeric"})
    uses_calc_time = True
    description_template = "the time since the first {}"

    def __init__(self, unit="seconds"):
        self.unit = unit.lower()

    def get_function(self, agg_type=Library.PANDAS):
        def time_since_first(values, time=None):
            time_since = time - values.iloc[0]
            return convert_time_units(time_since.total_seconds(), self.unit)

        return time_since_first


class Trend(AggregationPrimitive):
    """Calculates the trend of a column over time.

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
    input_types = [
        ColumnSchema(semantic_tags={"numeric"}),
        ColumnSchema(logical_type=Datetime, semantic_tags={"time_index"}),
    ]
    return_type = ColumnSchema(semantic_tags={"numeric"})
    description_template = "the linear trend of {} over time"

    def get_function(self, agg_type=Library.PANDAS):
        def pd_trend(y, x):

            return calculate_trend(pd.Series(data=y.values, index=x.values))

        return pd_trend


class Entropy(AggregationPrimitive):
    """Calculates the entropy for a categorical column

    Description:
        Given a list of observations from a categorical
        column return the entropy of the distribution.
        NaN values can be treated as a category or
        dropped.

    Args:
        dropna (bool): Whether to consider NaN values as a separate category
            Defaults to False.
        base (float): The logarithmic base to use
            Defaults to e (natural logarithm)

    Examples:
        >>> pd_entropy = Entropy()
        >>> pd_entropy([1,2,3,4])
        1.3862943611198906
    """

    name = "entropy"
    input_types = [ColumnSchema(semantic_tags={"category"})]
    return_type = ColumnSchema(semantic_tags={"numeric"})
    stack_on_self = False
    description_template = "the entropy of {}"

    def __init__(self, dropna=False, base=None):
        self.dropna = dropna
        self.base = base

    def get_function(self, agg_type=Library.PANDAS):
        def pd_entropy(s):
            distribution = s.value_counts(normalize=True, dropna=self.dropna)
            return stats.entropy(distribution, base=self.base)

        return pd_entropy
