from __future__ import division

from builtins import str

import numpy as np
import pandas as pd

from ..base.transform_primitive_base import (
    TransformPrimitive,
    make_trans_primitive
)

from featuretools.variable_types import (
    Boolean,
    Datetime,
    DatetimeTimeIndex,
    Discrete,
    Id,
    LatLong,
    Numeric,
    Ordinal,
    Text,
    Timedelta,
    Variable
)


class IsNull(TransformPrimitive):
    """For each value of base feature, return 'True' if value is null."""
    name = "is_null"
    input_types = [Variable]
    return_type = Boolean

    def get_function(self):
        return lambda array: pd.isnull(pd.Series(array))


class Absolute(TransformPrimitive):
    """Absolute value of base feature."""
    name = "absolute"
    input_types = [Numeric]
    return_type = Numeric

    def get_function(self):
        return lambda array: np.absolute(array)


class TimeSincePrevious(TransformPrimitive):
    """Compute the time since the previous instance."""
    name = "time_since_previous"
    input_types = [DatetimeTimeIndex, Id]
    return_type = Numeric

    def __init__(self, time_index, group_feature):
        """Summary

        Args:
            base_feature (PrimitiveBase): Base feature.
            group_feature (None, optional): Variable or feature to group
                rows by before calculating diff.

        """
        group_feature = self._check_feature(group_feature)
        assert issubclass(group_feature.variable_type, Discrete), \
            "group_feature must have a discrete variable_type"
        self.group_feature = group_feature
        super(TimeSincePrevious, self).__init__(time_index, group_feature)

    def generate_name(self):
        return u"time_since_previous_by_%s" % self.group_feature.get_name()

    def get_function(self):
        def pd_diff(base_array, group_array):
            bf_name = 'base_feature'
            groupby = 'groupby'
            grouped_df = pd.DataFrame.from_dict({bf_name: base_array,
                                                 groupby: group_array})
            grouped_df = grouped_df.groupby(groupby).diff()
            return grouped_df[bf_name].apply(lambda x: x.total_seconds())
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
     (seconds/days/etc) they encompass."""
    name = None
    input_types = [Timedelta]
    return_type = Numeric

    def get_function(self):
        return lambda array: pd_time_unit(self.name)(pd.TimedeltaIndex(array))


class Day(DatetimeUnitBasePrimitive):
    """Transform a Datetime feature into the day."""
    name = "day"


class Days(TimedeltaUnitBasePrimitive):
    """Transform a Timedelta feature into the number of days."""
    name = "days"


class Hour(DatetimeUnitBasePrimitive):
    """Transform a Datetime feature into the hour."""
    name = "hour"


class Hours(TimedeltaUnitBasePrimitive):
    """Transform a Timedelta feature into the number of hours."""
    name = "hours"

    def get_function(self):
        def pd_hours(array):
            return pd_time_unit("seconds")(pd.TimedeltaIndex(array)) / 3600.
        return pd_hours


class Second(DatetimeUnitBasePrimitive):
    """Transform a Datetime feature into the second."""
    name = "second"


class Seconds(TimedeltaUnitBasePrimitive):
    """Transform a Timedelta feature into the number of seconds."""
    name = "seconds"


class Minute(DatetimeUnitBasePrimitive):
    """Transform a Datetime feature into the minute."""
    name = "minute"


class Minutes(TimedeltaUnitBasePrimitive):
    """Transform a Timedelta feature into the number of minutes."""
    name = "minutes"

    def get_function(self):
        def pd_minutes(array):
            return pd_time_unit("seconds")(pd.TimedeltaIndex(array)) / 60
        return pd_minutes


class Week(DatetimeUnitBasePrimitive):
    """Transform a Datetime feature into the week."""
    name = "week"


class Weeks(TimedeltaUnitBasePrimitive):
    """Transform a Timedelta feature into the number of weeks."""
    name = "weeks"

    def get_function(self):
        def pd_weeks(array):
            return pd_time_unit("days")(pd.TimedeltaIndex(array)) / 7
        return pd_weeks


class Month(DatetimeUnitBasePrimitive):
    """Transform a Datetime feature into the month."""
    name = "month"


class Months(TimedeltaUnitBasePrimitive):
    """Transform a Timedelta feature into the number of months."""
    name = "months"

    def get_function(self):
        def pd_months(array):
            return pd_time_unit("days")(pd.TimedeltaIndex(array)) * (12 / 365)
        return pd_months


class Year(DatetimeUnitBasePrimitive):
    """Transform a Datetime feature into the year."""
    name = "year"


class Years(TimedeltaUnitBasePrimitive):
    """Transform a Timedelta feature into the number of years."""
    name = "years"

    def get_function(self):
        def pd_years(array):
            return pd_time_unit("days")(pd.TimedeltaIndex(array)) / 365
        return pd_years


class Weekend(TransformPrimitive):
    """Transform Datetime feature into the boolean of Weekend."""
    name = "weekend"
    input_types = [Datetime]
    return_type = Boolean

    def get_function(self):
        return lambda df: pd_time_unit("weekday")(pd.DatetimeIndex(df)) > 4


class Weekday(DatetimeUnitBasePrimitive):
    """Transform Datetime feature into the boolean of Weekday."""
    name = "weekday"


class NumCharacters(TransformPrimitive):
    """Return the characters in a given string.
    """
    name = 'characters'
    input_types = [Text]
    return_type = Numeric

    def get_function(self):
        return lambda array: pd.Series(array).fillna('').str.len()


class NumWords(TransformPrimitive):
    """Returns the words in a given string by counting the spaces.
    """
    name = 'numwords'
    input_types = [Text]
    return_type = Numeric

    def get_function(self):
        def word_counter(array):
            return pd.Series(array).fillna('').str.count(' ') + 1
        return word_counter


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


def pd_time_since(array, time):
    return (time - pd.DatetimeIndex(array)).values


TimeSince = make_trans_primitive(function=pd_time_since,
                                 input_types=[[DatetimeTimeIndex], [Datetime]],
                                 return_type=Timedelta,
                                 uses_calc_time=True,
                                 description="Calculates time since the cutoff time.",
                                 name="time_since")


class DaysSince(TransformPrimitive):
    """For each value of the base feature, compute the number of days between it
    and a datetime.
    """
    name = "days_since"
    input_types = [DatetimeTimeIndex]
    return_type = Numeric
    uses_calc_time = True

    def get_function(self):
        def pd_days_since(array, time):
            return pd_time_unit('days')(time - pd.DatetimeIndex(array))
        return pd_days_since


class IsIn(TransformPrimitive):
    """For each value of the base feature, checks whether it is in a provided list.
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

    def generate_name(self):
        return u"%s.isin(%s)" % (self.base_features[0].get_name(),
                                 str(self.list_of_outputs))


class Diff(TransformPrimitive):
    """Compute the difference between the value of a base feature and the previous value.

    If it is a Datetime feature, compute the difference in seconds.
    """
    name = "diff"
    input_types = [Numeric, Id]
    return_type = Numeric

    def __init__(self, base_feature, group_feature):
        """Summary

        Args:
            base_feature (PrimitiveBase): Base feature.
            group_feature (PrimitiveBase): Variable or feature to
                group rows by before calculating diff.

        """
        self.group_feature = self._check_feature(group_feature)
        super(Diff, self).__init__(base_feature, group_feature)

    def generate_name(self):
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
            try:
                return grouped_df[bf_name]
            except KeyError:
                return pd.Series([np.nan] * len(base_array))
        return pd_diff


class Not(TransformPrimitive):
    """For each value of the base feature, negates the boolean value.
    """
    name = "not"
    input_types = [Boolean]
    return_type = Boolean

    def generate_name(self):
        return u"NOT({})".format(self.base_features[0].get_name())

    def _get_op(self):
        return "__not__"

    def get_function(self):
        return lambda array: np.logical_not(array)


class Percentile(TransformPrimitive):
    """For each value of the base feature, determines the percentile in relation
    to the rest of the feature.
    """
    name = 'percentile'
    uses_full_entity = True
    input_types = [Numeric]
    return_type = Numeric

    def get_function(self):
        return lambda array: pd.Series(array).rank(pct=True)


def pd_time_unit(time_unit):
    def inner(pd_index):
        return getattr(pd_index, time_unit).values
    return inner


class Latitude(TransformPrimitive):
    """Returns the first value of the tuple base feature.
       For use with the LatLong variable type.
    """
    name = 'latitude'
    input_types = [LatLong]
    return_type = Numeric

    def get_function(self):
        return lambda array: pd.Series([x[0] for x in array])


class Longitude(TransformPrimitive):
    """Returns the second value on the tuple base feature.
       For use with the LatLong variable type.
    """
    name = 'longitude'
    input_types = [LatLong]
    return_type = Numeric

    def get_function(self):
        return lambda array: pd.Series([x[1] for x in array])


class Haversine(TransformPrimitive):
    """Calculate the approximate haversine distance in miles between two LatLong variable types.
    """
    name = 'haversine'
    input_types = [LatLong, LatLong]
    return_type = Numeric
    commutative = True

    def get_function(self):
        def haversine(latlong1, latlong2):
            lat_1s = np.array([x[0] for x in latlong1])
            lon_1s = np.array([x[1] for x in latlong1])
            lat_2s = np.array([x[0] for x in latlong2])
            lon_2s = np.array([x[1] for x in latlong2])
            lon1, lat1, lon2, lat2 = map(
                np.radians, [lon_1s, lat_1s, lon_2s, lat_2s])
            dlon = lon2 - lon1
            dlat = lat2 - lat1
            a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * \
                np.cos(lat2) * np.sin(dlon / 2.0)**2
            mi = 3950 * 2 * np.arcsin(np.sqrt(a))
            return mi
        return haversine
