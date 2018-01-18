from __future__ import division

from builtins import range, str
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from scipy.stats import skew

from .aggregation_primitive_base import (
    AggregationPrimitive,
    make_agg_primitive
)

from featuretools.variable_types import (
    Boolean,
    DatetimeTimeIndex,
    Discrete,
    Index,
    Numeric,
    Variable
)

# TODO: make sure get func gets numpy arrays not series


class Count(AggregationPrimitive):
    """Counts the number of non null values"""
    name = "count"
    input_types = [[Index], [Variable]]
    return_type = Numeric
    stack_on_self = False
    default_value = 0

    def __init__(self, id_feature, parent_entity, count_null=False, **kwargs):
        self.count_null = count_null
        super(Count, self).__init__(id_feature, parent_entity, **kwargs)

    def get_function(self):
        def func(values, count_null=self.count_null):
            if len(values) == 0:
                return 0

            if count_null:
                values = values.fillna(0)

            return values.count()
        return func

    def generate_name(self):
        where_str = self._where_str()
        use_prev_str = self._use_prev_str()

        return u"COUNT(%s%s%s)" % (self.child_entity.name,
                                   where_str, use_prev_str)


class Sum(AggregationPrimitive):
    """Counts the number of elements of a numeric or boolean feature"""
    name = "sum"
    input_types = [Numeric]
    return_type = Numeric
    stack_on_self = False
    stack_on_exclude = [Count]

    # todo: handle count nulls
    def get_function(self):
        def sum_func(x):
            return np.nan_to_num(x.values).sum(dtype=np.float)
        return sum_func


class Mean(AggregationPrimitive):
    """Computes the average value of a numeric feature"""
    name = "mean"
    input_types = [Numeric]
    return_type = Numeric

    # p todo: handle nulls
    def get_function(self):
        return np.nanmean


class Mode(AggregationPrimitive):
    """Finds the most common element in a categorical feature"""
    name = "mode"
    input_types = [Discrete]
    return_type = None

    def get_function(self):
        def pd_mode(x):
            if x.mode().shape[0] == 0:
                return np.nan
            return x.mode().iloc[0]
        return pd_mode


Min = make_agg_primitive(
    np.min,
    [Numeric],
    None,
    name="min",
    stack_on_self=False,
    description="Finds the minimum non-null value of a numeric feature.")


# class Min(AggregationPrimitive):
#     """Finds the minimum non-null value of a numeric feature."""
#     name = "min"
#     input_types =  [Numeric]
#     return_type = None
#     # max_stack_depth = 1
#     stack_on_self = False

#     def get_function(self):
#         return np.min


class Max(AggregationPrimitive):
    """Finds the maximum non-null value of a numeric feature"""
    name = "max"
    input_types = [Numeric]
    return_type = None
    # max_stack_depth = 1
    stack_on_self = False

    def get_function(self):
        return np.max


class NUnique(AggregationPrimitive):
    """Returns the number of unique categorical variables"""
    name = "num_unique"
    # todo can we use discrete in input_types instead?
    input_types = [Discrete]
    return_type = Numeric
    # max_stack_depth = 1
    stack_on_self = False

    def get_function(self):
        return lambda x: x.nunique()


class NumTrue(AggregationPrimitive):
    """Finds the number of 'True' values in a boolean"""
    name = "num_true"
    input_types = [Boolean]
    return_type = Numeric
    default_value = 0
    stack_on = []
    stack_on_exclude = []

    def get_function(self):
        def num_true(x):
            return np.nan_to_num(x.values).sum()
        return num_true


class PercentTrue(AggregationPrimitive):
    """Finds the percent of 'True' values in a boolean feature"""
    name = "percent_true"
    input_types = [Boolean]
    return_type = Numeric
    max_stack_depth = 1
    stack_on = []
    stack_on_exclude = []

    def get_function(self):
        def percent_true(x):
            if len(x) == 0:
                return np.nan
            return np.nan_to_num(x.values).sum(dtype=np.float) / len(x)
        return percent_true


class NMostCommon(AggregationPrimitive):
    """Finds the N most common elements in a categorical feature"""
    name = "n_most_common"
    input_types = [Discrete]
    return_type = Discrete
    # max_stack_depth = 1
    stack_on = []
    stack_on_exclude = []
    expanding = True

    def __init__(self, base_feature, parent_entity, n=3):
        self.n = n
        super(NMostCommon, self).__init__(base_feature, parent_entity)

    @property
    def default_value(self):
        return np.zeros(self.n) * np.nan

    def get_expanded_names(self):
        names = []
        for i in range(1, self.n + 1):
            names.append(str(i) + self.get_name()[1:])
        return names

    def get_function(self):
        def pd_topn(x, n=self.n):
            return np.array(x.value_counts()[:n].index)
        return pd_topn


class AvgTimeBetween(AggregationPrimitive):
    """Computes the average time between consecutive events
    using the time index of the entity.

    Note: equivalent to Mean(Diff(time_index)), but more performant
    """

    # Potentially unnecessary if we add an trans_feat that
    # calculates the difference between events. DFS
    # should then calculate the average of that trans_feat
    # which amounts to AvgTimeBetween
    name = "avg_time_between"
    input_types = [DatetimeTimeIndex]
    return_type = Numeric
    # max_stack_depth = 1

    def get_function(self):
        def pd_avg_time_between(x):
            """
            Assumes time scales are closer to order
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
    """Finds the median value of any feature with well-ordered values"""
    name = "median"
    input_types = [Numeric]
    return_type = None
    # max_stack_depth = 2

    def get_function(self):
        return lambda x: x.median()


class Skew(AggregationPrimitive):
    """Computes the skewness of a data set.

    For normally distributed data, the skewness should be about 0. A skewness
    value > 0 means that there is more weight in the left tail of the
    distribution.
    """
    name = "skew"
    input_types = [Numeric]
    return_type = Numeric
    stack_on = []
    stack_on_self = False
    # max_stack_depth = 1

    def get_function(self):
        return skew


class Std(AggregationPrimitive):
    """
    Finds the standard deviation of a numeric feature ignoring null values.
    """
    name = "std"
    input_types = [Numeric]
    return_type = Numeric
    # max_stack_depth = 2
    stack_on_self = False

    def get_function(self):
        return np.nanstd


class Last(AggregationPrimitive):
    """Returns the last value"""
    name = "last"
    input_types = [Variable]
    return_type = None
    stack_on_self = False
    # max_stack_depth = 1

    def get_function(self):
        def pd_last(x):
            return x.iloc[-1]
        return pd_last


class Any(AggregationPrimitive):
    """Test if any value is 'True'"""
    name = "any"
    input_types = [Boolean]
    return_type = Boolean
    stack_on_self = False

    def get_function(self):
        return np.any


class All(AggregationPrimitive):
    """Test if all values are 'True'"""
    name = "all"
    input_types = [Boolean]
    return_type = Boolean
    stack_on_self = False

    def get_function(self):
        return np.all


class TimeSinceLast(AggregationPrimitive):
    """Time since last related instance"""
    name = "time_since_last"
    input_types = [DatetimeTimeIndex]
    return_type = Numeric
    uses_calc_time = True

    def get_function(self):

        def time_since_last(values, time=None):
            time_since = time - values.iloc[0]
            return time_since.total_seconds()

        return time_since_last


class Trend(AggregationPrimitive):
    """Calculates the slope of the linear trend of variable overtime"""
    name = "trend"
    input_types = [Numeric, DatetimeTimeIndex]
    return_type = Numeric

    def __init__(self, base_features, parent_entity, **kwargs):
        self.value = base_features[0]
        self.time_index = base_features[1]
        super(Trend, self).__init__(base_features,
                                    parent_entity,
                                    **kwargs)

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


# # TODO: Not implemented yet
# class ConseqPos(AggregationPrimitive):
#     name = "conseq_pos"
#     input_types =  [(variable_types.Numeric,),
#                 (variable_types.Ordinal,)]
#     return_type = variable_types.Numeric
#     max_stack_depth = 1
#     stack_on = []
#     stack_on_exclude = []

#     def get_function(self):
#         raise NotImplementedError("This feature has not been implemented")


# # TODO: Not implemented yet
# class ConseqSame(AggregationPrimitive):
#     name = "conseq_same"
#     input_types =  [(variable_types.Categorical,),
#                 (variable_types.Ordinal,),
#                 (variable_types.Numeric,)]
#     return_type = variable_types.Numeric
#     max_stack_depth = 1
#     stack_on = []
#     stack_on_exclude = []

#     def get_function(self):
#         raise NotImplementedError("This feature has not been implemented")


# # TODO: Not implemented yet
# class TimeSinceLast(AggregationPrimitive):


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
    """
    Finds whether time best corresponds to a value in
    days, hours, minutes, or seconds
    """
    for dividend in [86400, 3600, 60]:
        div = time / dividend
        if round(div) == div:
            return dividend
    return 1
