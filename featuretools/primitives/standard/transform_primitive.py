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
        return np.absolute


class TimeSincePrevious(TransformPrimitive):
    """Compute the time since the previous instance."""
    name = "time_since_previous"
    input_types = [DatetimeTimeIndex, Id]
    return_type = Numeric

    def generate_name(self, base_feature_names):
        return u"time_since_previous_by_%s" % base_feature_names[1]

    def get_function(self):
        def pd_diff(base_array, group_array):
            bf_name = 'base_feature'
            groupby = 'groupby'
            grouped_df = pd.DataFrame.from_dict({bf_name: base_array,
                                                 groupby: group_array})
            grouped_df = grouped_df.groupby(groupby).diff()
            return grouped_df[bf_name].apply(lambda x: x.total_seconds())
        return pd_diff


class Day(TransformPrimitive):
    """Transform a Datetime feature into the day."""
    name = "day"
    input_types = [Datetime]
    return_type = Ordinal

    def get_function(self):
        def day(vals):
            return pd.DatetimeIndex(vals).day.values
        return day


class Hour(TransformPrimitive):
    """Transform a Datetime feature into the hour."""
    name = "hour"
    input_types = [Datetime]
    return_type = Ordinal

    def get_function(self):
        def hour(vals):
            return pd.DatetimeIndex(vals).hour.values
        return hour


class Second(TransformPrimitive):
    """Transform a Datetime feature into the second."""
    name = "second"
    input_types = [Datetime]
    return_type = Numeric

    def get_function(self):
        def second(vals):
            return pd.DatetimeIndex(vals).second.values
        return second


class Minute(TransformPrimitive):
    """Transform a Datetime feature into the "minute."""
    name = "minute"
    input_types = [Datetime]
    return_type = Numeric

    def get_function(self):
        def minute(vals):
            return pd.DatetimeIndex(vals).minute.values
        return minute


class Week(TransformPrimitive):
    """Transform a Datetime feature into the week."""
    name = "week"
    input_types = [Datetime]
    return_type = Ordinal

    def get_function(self):
        def week(vals):
            return pd.DatetimeIndex(vals).week.values
        return week


class Month(TransformPrimitive):
    """Transform a Datetime feature into the  "month."""
    name = "month"
    input_types = [Datetime]
    return_type = Ordinal

    def get_function(self):
        def month(vals):
            return pd.DatetimeIndex(vals).month.values
        return month


class Year(TransformPrimitive):
    """Transform a Datetime feature into the year."""
    name = "year"
    input_types = [Datetime]
    return_type = Ordinal

    def get_function(self):
        def year(vals):
            return pd.DatetimeIndex(vals).year.values
        return year


class IsWeekend(TransformPrimitive):
    """Transform Datetime feature into the boolean of Weekend."""
    name = "is_weekend"
    input_types = [Datetime]
    return_type = Boolean

    def get_function(self):
        def is_weekend(vals):
            return pd.DatetimeIndex(vals).weekday.values > 4
        return is_weekend


class Weekday(TransformPrimitive):
    """Transform a Datetime feature into the weekday."""
    name = "weekday"
    input_types = [Datetime]
    return_type = Ordinal

    def get_function(self):
        def weekday(vals):
            return pd.DatetimeIndex(vals).weekday.values
        return weekday


class NumCharacters(TransformPrimitive):
    """Return the number of characters in a given string.
    """
    name = 'num_characters'
    input_types = [Text]
    return_type = Numeric

    def get_function(self):
        return lambda array: pd.Series(array).fillna('').str.len()


class NumWords(TransformPrimitive):
    """Returns the number of words in a given string by counting the spaces.
    """
    name = 'num_words'
    input_types = [Text]
    return_type = Numeric

    def get_function(self):
        def word_counter(array):
            return pd.Series(array).fillna('').str.count(' ') + 1
        return word_counter


def pd_time_since(array, time):
    """Calculates time since the cutoff time."""
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
            return (time - pd.DatetimeIndex(array)).days
        return pd_days_since


class IsIn(TransformPrimitive):
    """For each value of the base feature, checks whether it is in a provided list.
    """
    name = "isin"
    input_types = [Variable]
    return_type = Boolean

    def __init__(self, list_of_outputs=None):
        self.list_of_outputs = list_of_outputs

    def get_function(self):
        def pd_is_in(array):
            return pd.Series(array).isin(self.list_of_outputs or [])
        return pd_is_in

    def generate_name(self, base_feature_names):
        return u"%s.isin(%s)" % (base_feature_names[0],
                                 str(self.list_of_outputs))


class Diff(TransformPrimitive):
    """Compute the difference between the value of a base feature and the previous value.

    If it is a Datetime feature, compute the difference in seconds.
    """
    name = "diff"
    input_types = [Numeric, Id]
    return_type = Numeric

    def generate_name(self, base_feature_names):
        base_features_str = base_feature_names[0] + u" by " + \
            base_feature_names[1]
        return u"DIFF(%s)" % (base_features_str)

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


class Negate(TransformPrimitive):
    name = "negate"
    input_types = [Numeric]
    return_type = Numeric

    def get_function(self):
        def negate(vals):
            return vals * -1
        return negate

    def generate_name(self, base_feature_names):
        return "-(%s)" % (base_feature_names[0])


class Not(TransformPrimitive):
    """For each value of the base feature, negates the boolean value."""
    name = "not"
    input_types = [Boolean]
    return_type = Boolean

    def generate_name(self, base_feature_names):
        return u"NOT({})".format(base_feature_names[0])

    def get_function(self):
        return np.logical_not


class Percentile(TransformPrimitive):
    """For each value of the base feature, determines the percentile in
        relation to the rest of the feature.
    """
    name = 'percentile'
    uses_full_entity = True
    input_types = [Numeric]
    return_type = Numeric

    def get_function(self):
        return lambda array: pd.Series(array).rank(pct=True)


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
    """Calculate the approximate haversine distance between two LatLong
        variable types. Defaults to computing in miles.

        Args:
            unit (str): Determines the unit value to output. Could
                be `miles` or `kilometers`. Default is `miles.

        Example:

           .. code-block:: python

                from featuretools.primitives import Haversine
                haversine_miles = Haversine(unit='miles')

    """
    name = 'haversine'
    input_types = [LatLong, LatLong]
    return_type = Numeric
    commutative = True

    def __init__(self, unit='miles'):
        valid_units = ['miles', 'kilometers']
        if unit not in valid_units:
            error_message = 'Invalid unit %s provided. Must be one of %s' % (unit, valid_units)
            raise ValueError(error_message)
        self.unit = unit

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
            radius_earth = 3958.7613
            if self.unit == 'kilometers':
                radius_earth = 6371.0088
            distance = radius_earth * 2 * np.arcsin(np.sqrt(a))
            return distance
        return haversine

    def generate_name(self, base_feature_names):
        name = u"{}(".format(self.name.upper())
        name += u", ".join(base_feature_names)
        if self.unit != 'miles':
            name += u", unit={}".format(self.unit)
        name += u")"
        return name
