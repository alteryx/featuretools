import datetime
import functools
import os

import numpy as np
import pandas as pd

from .primitive_base import PrimitiveBase
from .utils import inspect_function_args

from featuretools.variable_types import (Boolean, Datetime, DatetimeTimeIndex,
                                         Discrete, Id, Numeric, Ordinal,
                                         Timedelta, Variable)

current_path = os.path.dirname(os.path.realpath(__file__))
FEATURE_DATASETS = os.path.join(os.path.join(current_path, '..'),
                                'feature_datasets')


class TransformPrimitive(PrimitiveBase):
    """Feature for entity that is a based off one or more other features
        in that entity"""
    rolling_function = False

    def __init__(self, *base_features):
        # Any edits made to this method should also be made to the
        # new_class_init method in make_trans_primitive
        self.base_features = [self._check_feature(f) for f in base_features]
        if any(bf.expanding for bf in self.base_features):
            self.expanding = True
        assert len(set([f.entity for f in self.base_features])) == 1, \
            "More than one entity for base features"
        super(TransformPrimitive, self).__init__(self.base_features[0].entity,
                                                 self.base_features)

    def _get_name(self):
        name = u"{}(".format(self.name.upper())
        name += u", ".join(f.get_name() for f in self.base_features)
        name += u")"
        return name

    @property
    def default_value(self):
        return self.base_features[0].default_value


def make_trans_primitive(function, input_types, return_type, name=None,
                         description='A custom transform primitive',
                         cls_attributes=None, uses_calc_time=False,
                         associative=False):
    '''Returns a new transform primitive class

    Args:
        function (function): function that takes in an array  and applies some
            transformation to it.

        name (string): name of the function

        input_types (list[:class:`.Variable`]): variable types of the inputs

        return_type (:class:`.Variable`): variable type of return

        description (string): description of primitive

        cls_attributes (dict): custom attributes to be added to class

        uses_calc_time (bool): if True, the cutoff time the feature is being
            calculated at will be passed to the function as the keyword
            argument 'time'.

        associative (bool): If True, will only make one feature per unique set
            of base features

    Example:
        .. ipython :: python

            from featuretools.primitives import make_trans_primitive
            from featuretools.variable_types import Variable, Boolean


            def pd_is_in(array, list_of_outputs=None):
                if list_of_outputs is None:
                    list_of_outputs = []
                return pd.Series(array).isin(list_of_outputs)

            def isin_get_name(self):
                return u"%s.isin(%s)" % (self.base_features[0].get_name(),
                                         str(self.kwargs['list_of_outputs']))

            IsIn = make_trans_primitive(
                pd_is_in,
                [Variable],
                Boolean,
                name="is_in",
                description="For each value of the base feature, checks "
                "whether it is in a list that provided.",
                cls_attributes={"_get_name": isin_get_name})
    '''
    # dictionary that holds attributes for class
    cls = {"__doc__": description}
    if cls_attributes is not None:
        cls.update(cls_attributes)

    # creates the new class and set name and types
    name = name or function.func_name
    new_class = type(name, (TransformPrimitive,), cls)
    new_class.name = name
    new_class.input_types = input_types
    new_class.return_type = return_type
    new_class.associative = associative
    new_class, kwargs = inspect_function_args(new_class,
                                              function,
                                              uses_calc_time)

    if kwargs is not None:
        new_class.kwargs = kwargs

        def new_class_init(self, *args, **kwargs):
            self.base_features = [self._check_feature(f) for f in args]
            if any(bf.expanding for bf in self.base_features):
                self.expanding = True
            assert len(set([f.entity for f in self.base_features])) == 1, \
                "More than one entity for base features"
            self.kwargs.update(kwargs)
            self.partial = functools.partial(function, **self.kwargs)
            super(TransformPrimitive, self).__init__(
                self.base_features[0].entity, self.base_features)
        new_class.__init__ = new_class_init
        new_class.get_function = lambda self: self.partial
    else:
        # creates a lambda function that returns function every time
        new_class.get_function = lambda self, f=function: f

    return new_class


class IsNull(TransformPrimitive):
    """For each value of base feature, return true if value is null"""
    name = "is_null"
    input_types = [Variable]
    return_type = Boolean

    def get_function(self):
        return lambda array: pd.isnull(pd.Series(array))


class Absolute(TransformPrimitive):
    """Absolute value of base feature"""
    name = "absolute"
    input_types = [Numeric]
    return_type = Numeric

    def get_function(self):
        return lambda array: np.absolute(array)


class TimeSincePrevious(TransformPrimitive):
    """Compute the time since the previous instance for each instance in a
     time indexed entity"""
    name = "time_since_previous"
    input_types = [DatetimeTimeIndex, Id]
    return_type = Numeric

    def __init__(self, time_index, group_feature):
        """Summary

        Args:
            base_feature (:class:`PrimitiveBase`): base feature
            group_feature (None, optional): variable or feature to group
                rows by before calculating diff

        """
        group_feature = self._check_feature(group_feature)
        assert issubclass(group_feature.variable_type, Discrete), \
            "group_feature must have a discrete variable_type"
        self.group_feature = group_feature
        super(TimeSincePrevious, self).__init__(time_index, group_feature)

    def _get_name(self):
        return u"time_since_previous_by_%s" % self.group_feature.get_name()

    def get_function(self):
        def pd_diff(base_array, group_array):
            bf_name = 'base_feature'
            groupby = 'groupby'
            grouped_df = pd.DataFrame.from_dict({bf_name: base_array,
                                                 groupby: group_array})
            grouped_df = grouped_df.groupby(groupby).diff()
            return grouped_df[bf_name].apply(lambda x:
                                             x.total_seconds())
        return pd_diff


class DatetimeUnitBasePrimitive(TransformPrimitive):
    """Transform Datetime feature into time or calendar units
     (second/day/week/etc)"""
    name = None
    input_types = [Datetime]
    return_type = Ordinal

    def get_function(self):
        return lambda array: pd_time_unit(self.name)(pd.DatetimeIndex(array))


class TimedeltaUnitBasePrimitive(TransformPrimitive):
    """Transform Timedelta features into number of time units
     (seconds/days/etc) they encompass"""
    name = None
    input_types = [Timedelta]
    return_type = Numeric

    def get_function(self):
        return lambda array: pd_time_unit(self.name)(pd.TimedeltaIndex(array))


class Day(DatetimeUnitBasePrimitive):
    """Transform a Datetime feature into the day"""
    name = "day"


class Days(TimedeltaUnitBasePrimitive):
    """Transform a Timedelta feature into the number of days"""
    name = "days"


class Hour(DatetimeUnitBasePrimitive):
    """Transform a Datetime feature into the hour"""
    name = "hour"


class Hours(TimedeltaUnitBasePrimitive):
    """Transform a Timedelta feature into the number of hours"""
    name = "hours"

    def get_function(self):
        def pd_hours(array):
            return pd_time_unit("seconds")(pd.TimedeltaIndex(array)) / 3600.
        return pd_hours


class Second(DatetimeUnitBasePrimitive):
    """Transform a Datetime feature into the second"""
    name = "second"


class Seconds(TimedeltaUnitBasePrimitive):
    """Transform a Timedelta feature into the number of seconds"""
    name = "seconds"


class Minute(DatetimeUnitBasePrimitive):
    """Transform a Datetime feature into the minute"""
    name = "minute"


class Minutes(TimedeltaUnitBasePrimitive):
    """Transform a Timedelta feature into the number of minutes"""
    name = "minutes"

    def get_function(self):
        def pd_minutes(array):
            return pd_time_unit("seconds")(pd.TimedeltaIndex(array)) / 60.
        return pd_minutes


class Week(DatetimeUnitBasePrimitive):
    """Transform a Datetime feature into the week"""
    name = "week"


class Weeks(TimedeltaUnitBasePrimitive):
    """Transform a Timedelta feature into the number of weeks"""
    name = "weeks"

    def get_function(self):
        def pd_weeks(array):
            return pd_time_unit("days")(pd.TimedeltaIndex(array)) / 7.
        return pd_weeks


class Month(DatetimeUnitBasePrimitive):
    """Transform a Datetime feature into the month"""
    name = "month"


class Months(TimedeltaUnitBasePrimitive):
    """Transform a Timedelta feature into the number of months"""
    name = "months"

    def get_function(self):
        def pd_months(array):
            return pd_time_unit("days")(pd.TimedeltaIndex(array)) * (12. / 365)
        return pd_months


class Year(DatetimeUnitBasePrimitive):
    """Transform a Datetime feature into the year"""
    name = "year"


class Years(TimedeltaUnitBasePrimitive):
    """Transform a Timedelta feature into the number of years"""
    name = "years"

    def get_function(self):
        def pd_years(array):
            return pd_time_unit("days")(pd.TimedeltaIndex(array)) / 365
        return pd_years


class Weekend(TransformPrimitive):
    """Transform Datetime feature into the boolean of Weekend"""
    name = "is_weekend"
    input_types = [Datetime]
    return_type = Boolean

    def get_function(self):
        return lambda df: pd_time_unit("weekday")(pd.DatetimeIndex(df)) > 4


class Weekday(DatetimeUnitBasePrimitive):
    """Transform Datetime feature into the boolean of Weekday"""
    name = "weekday"


# class Like(TransformPrimitive):
#     """Equivalent to SQL LIKE(%text%)
#        Returns true if text is contained with the string base_feature
#     """
#     name = "like"
#     input_types =  [(Text,), (Categorical,)]
#     return_type = Boolean

#     def __init__(self, base_feature, like_statement, case_sensitive=False):
#         self.like_statement = like_statement
#         self.case_sensitive = case_sensitive
#         super(Like, self).__init__(base_feature)

#     def get_function(self):
#         def pd_like(df, f):
#             return df[df.columns[0]].str.contains(f.like_statement,
#                                                   case=f.case_sensitive)
#         return pd_like


# class TimeSince(TransformPrimitive):
#     """
#     For each value of the base feature, compute the timedelta between it and
#     a datetime
#     """
#     name = "time_since"
#     input_types = [[DatetimeTimeIndex], [Datetime]]
#     return_type = Timedelta
#     uses_calc_time = True

#     def get_function(self):
#         def pd_time_since(array, time):
#             if time is None:
#                 time = datetime.now()
#             return (time - pd.DatetimeIndex(array)).values
#         return pd_time_since


def pd_time_since(array, time):
    if time is None:
        time = datetime.now()
    return (time - pd.DatetimeIndex(array)).values


TimeSince = make_trans_primitive(function=pd_time_since,
                                 input_types=[[DatetimeTimeIndex], [Datetime]],
                                 return_type=Timedelta,
                                 uses_calc_time=True,
                                 name="time_since")


class DaysSince(TransformPrimitive):
    """
    For each value of the base feature, compute the number of days between it
    and a datetime
    """
    name = "days_since"
    input_types = [DatetimeTimeIndex]
    return_type = Numeric
    uses_calc_time = True

    def get_function(self):
        def pd_days_since(array, time):
            if time is None:
                time = datetime.now()
            return pd_time_unit('days')(time - pd.DatetimeIndex(array))
        return pd_days_since


class IsIn(TransformPrimitive):
    """
    For each value of the base feature, checks whether it is in a list that is
    provided.
    """
    name = "isin"
    input_types = [Variable]
    return_type = Boolean

    def __init__(self, base_feature, list_of_outputs=None):
        self.list_of_outputs = list_of_outputs
        super(IsIn, self).__init__(base_feature)

    def get_function(self):
        def pd_is_in(array, list_of_outputs=self.list_of_outputs):
            if list_of_outputs is None:
                list_of_outputs = []
            return pd.Series(array).isin(list_of_outputs)
        return pd_is_in

    def _get_name(self):
        return u"%s.isin(%s)" % (self.base_features[0].get_name(),
                                 str(self.list_of_outputs))


class Diff(TransformPrimitive):
    """
    For each value of the base feature, compute the difference between it and
    the previous value.

    If it is a Datetime feature, compute the difference in seconds
    """
    name = "diff"
    input_types = [Numeric, Id]
    return_type = Numeric

    def __init__(self, base_feature, group_feature):
        """Summary

        Args:
            base_feature (:class:`PrimitiveBase`): base feature
            group_feature (:class:`PrimitiveBase`): variable or feature to
                group rows by before calculating diff

        """
        self.group_feature = self._check_feature(group_feature)
        super(Diff, self).__init__(base_feature, group_feature)

    def _get_name(self):
        base_features_str = self.base_features[0].get_name() + u" by " + \
            self.group_feature.get_name()
        return u"%s(%s)" % (self.name.upper(), base_features_str)

    def get_function(self):
        def pd_diff(base_array, group_array):
            bf_name = 'base_feature'
            groupby = 'groupby'
            grouped_df = pd.DataFrame.from_dict({bf_name: base_array,
                                                 groupby: group_array})
            grouped_df = grouped_df.groupby(groupby).diff()
            return grouped_df[bf_name]
        return pd_diff


class Not(TransformPrimitive):
    """
    For each value of the base feature, negates the boolean value.
    """
    name = "not"
    input_types = [Boolean]
    return_type = Boolean

    def _get_name(self):
        return u"NOT({})".format(self.base_features[0].get_name())

    def _get_op(self):
        return "__not__"

    def get_function(self):
        return lambda array: np.logical_not(array)


class Percentile(TransformPrimitive):
    """
    For each value of the base feature, determines the percentile in relation
    to the rest of the feature.
    """
    name = 'percentile'
    input_types = [Numeric]
    return_type = Numeric

    def get_function(self):
        return lambda array: pd.Series(array).rank(pct=True)


def pd_time_unit(time_unit):
    def inner(pd_index):
        return getattr(pd_index, time_unit).values
    return inner
