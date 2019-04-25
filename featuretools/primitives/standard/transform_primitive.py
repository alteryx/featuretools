from __future__ import division

from builtins import str

import numpy as np
import pandas as pd

from ..base.transform_primitive_base import TransformPrimitive

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
    """Determines if a value is null.

    Examples:
        >>> is_null = IsNull()
        >>> is_null([1, None, 3]).tolist()
        [False, True, False]
    """
    name = "is_null"
    input_types = [Variable]
    return_type = Boolean

    def get_function(self):
        return lambda array: pd.isnull(pd.Series(array))


class Absolute(TransformPrimitive):
    """Computes the absolute value of a number.

    Examples:
        >>> absolute = Absolute()
        >>> absolute([3.0, -5.0, -2.4]).tolist()
        [3.0, 5.0, 2.4]
    """
    name = "absolute"
    input_types = [Numeric]
    return_type = Numeric

    def get_function(self):
        return np.absolute


class TimeSincePrevious(TransformPrimitive):
    """Compute the time in seconds since the previous instance of an entry.

    Description:
        Given a list of datetimes and a corresponding list of item ID values,
        compute the time in seconds elapsed since the previous occurrence
        of the item in the list. If an item is present only once, the result
        for this item will be `NaN`. Similarly, the result for the first
        occurrence of an item will always be `NaN`.

    Examples:
        >>> from datetime import datetime
        >>> time_since_previous = TimeSincePrevious()
        >>> dates = [datetime(2019, 3, 1, 0, 0, 0),
        ...          datetime(2019, 3, 1, 0, 2, 0),
        ...          datetime(2019, 3, 10, 0, 0, 0),
        ...          datetime(2019, 3, 1, 0, 2, 30),
        ...          datetime(2019, 3, 10, 0, 0, 50)]
        >>> labels = ['A', 'A', 'B', 'A', 'B']
        >>> time_since_previous(dates, labels).tolist()
        [nan, 120.0, nan, 30.0, 50.0]
    """
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
    """Determines the day of the month from a datetime.

    Examples:
        >>> from datetime import datetime
        >>> dates = [datetime(2019, 3, 1),
        ...          datetime(2019, 3, 3),
        ...          datetime(2019, 3, 31)]
        >>> day = Day()
        >>> day(dates).tolist()
        [1, 3, 31]
    """
    name = "day"
    input_types = [Datetime]
    return_type = Ordinal

    def get_function(self):
        def day(vals):
            return pd.DatetimeIndex(vals).day.values
        return day


class Hour(TransformPrimitive):
    """Determines the hour value of a datetime.

    Examples:
        >>> from datetime import datetime
        >>> dates = [datetime(2019, 3, 1),
        ...          datetime(2019, 3, 3, 11, 10, 50),
        ...          datetime(2019, 3, 31, 19, 45, 15)]
        >>> hour = Hour()
        >>> hour(dates).tolist()
        [0, 11, 19]
    """
    name = "hour"
    input_types = [Datetime]
    return_type = Ordinal

    def get_function(self):
        def hour(vals):
            return pd.DatetimeIndex(vals).hour.values
        return hour


class Second(TransformPrimitive):
    """Determines the seconds value of a datetime.

    Examples:
        >>> from datetime import datetime
        >>> dates = [datetime(2019, 3, 1),
        ...          datetime(2019, 3, 3, 11, 10, 50),
        ...          datetime(2019, 3, 31, 19, 45, 15)]
        >>> second = Second()
        >>> second(dates).tolist()
        [0, 50, 15]
    """
    name = "second"
    input_types = [Datetime]
    return_type = Numeric

    def get_function(self):
        def second(vals):
            return pd.DatetimeIndex(vals).second.values
        return second


class Minute(TransformPrimitive):
    """Determines the minutes value of a datetime.

    Examples:
        >>> from datetime import datetime
        >>> dates = [datetime(2019, 3, 1),
        ...          datetime(2019, 3, 3, 11, 10, 50),
        ...          datetime(2019, 3, 31, 19, 45, 15)]
        >>> minute = Minute()
        >>> minute(dates).tolist()
        [0, 10, 45]
    """
    name = "minute"
    input_types = [Datetime]
    return_type = Numeric

    def get_function(self):
        def minute(vals):
            return pd.DatetimeIndex(vals).minute.values
        return minute


class Week(TransformPrimitive):
    """Determines the week of the year from a datetime.

    Description:
        Returns the week of the year from a datetime value. The first week
        of the year starts on January 1, and week numbers increment each
        Monday.

    Examples:
        >>> from datetime import datetime
        >>> dates = [datetime(2019, 1, 3),
        ...          datetime(2019, 6, 17, 11, 10, 50),
        ...          datetime(2019, 11, 30, 19, 45, 15)]
        >>> week = Week()
        >>> week(dates).tolist()
        [1, 25, 48]
        """
    name = "week"
    input_types = [Datetime]
    return_type = Ordinal

    def get_function(self):
        def week(vals):
            return pd.DatetimeIndex(vals).week.values
        return week


class Month(TransformPrimitive):
    """Determines the month value of a datetime.

    Examples:
        >>> from datetime import datetime
        >>> dates = [datetime(2019, 3, 1),
        ...          datetime(2019, 6, 17, 11, 10, 50),
        ...          datetime(2019, 11, 30, 19, 45, 15)]
        >>> month = Month()
        >>> month(dates).tolist()
        [3, 6, 11]
    """
    name = "month"
    input_types = [Datetime]
    return_type = Ordinal

    def get_function(self):
        def month(vals):
            return pd.DatetimeIndex(vals).month.values
        return month


class Year(TransformPrimitive):
    """Determines the year value of a datetime.

    Examples:
        >>> from datetime import datetime
        >>> dates = [datetime(2019, 3, 1),
        ...          datetime(2048, 6, 17, 11, 10, 50),
        ...          datetime(1950, 11, 30, 19, 45, 15)]
        >>> year = Year()
        >>> year(dates).tolist()
        [2019, 2048, 1950]
    """
    name = "year"
    input_types = [Datetime]
    return_type = Ordinal

    def get_function(self):
        def year(vals):
            return pd.DatetimeIndex(vals).year.values
        return year


class IsWeekend(TransformPrimitive):
    """Determines if a date falls on a weekend.

    Examples:
        >>> from datetime import datetime
        >>> dates = [datetime(2019, 3, 1),
        ...          datetime(2019, 6, 17, 11, 10, 50),
        ...          datetime(2019, 11, 30, 19, 45, 15)]
        >>> is_weekend = IsWeekend()
        >>> is_weekend(dates).tolist()
        [False, False, True]
    """
    name = "is_weekend"
    input_types = [Datetime]
    return_type = Boolean

    def get_function(self):
        def is_weekend(vals):
            return pd.DatetimeIndex(vals).weekday.values > 4
        return is_weekend


class Weekday(TransformPrimitive):
    """Determines the day of the week from a datetime.

    Description:
        Returns the day of the week from a datetime value. Weeks
        start on Monday (day 0) and run through Sunday (day 6).

    Examples:
        >>> from datetime import datetime
        >>> dates = [datetime(2019, 3, 1),
        ...          datetime(2019, 6, 17, 11, 10, 50),
        ...          datetime(2019, 11, 30, 19, 45, 15)]
        >>> weekday = Weekday()
        >>> weekday(dates).tolist()
        [4, 0, 5]
    """
    name = "weekday"
    input_types = [Datetime]
    return_type = Ordinal

    def get_function(self):
        def weekday(vals):
            return pd.DatetimeIndex(vals).weekday.values
        return weekday


class NumCharacters(TransformPrimitive):
    """Calculates the number of characters in a string.

    Examples:
        >>> num_characters = NumCharacters()
        >>> num_characters(['This is a string',
        ...                 'second item',
        ...                 'final1']).tolist()
        [16, 11, 6]
    """
    name = 'num_characters'
    input_types = [Text]
    return_type = Numeric

    def get_function(self):
        return lambda array: pd.Series(array).fillna('').str.len()


class NumWords(TransformPrimitive):
    """Determines the number of words in a string by counting the spaces.

    Examples:
        >>> num_words = NumWords()
        >>> num_words(['This is a string',
        ...            'Two words',
        ...            'no-spaces',
        ...            'Also works with sentences. Second sentence!']).tolist()
        [4, 2, 1, 6]
    """
    name = 'num_words'
    input_types = [Text]
    return_type = Numeric

    def get_function(self):
        def word_counter(array):
            return pd.Series(array).fillna('').str.count(' ') + 1
        return word_counter


class TimeSince(TransformPrimitive):
    """Calculates time in nanoseconds from a value to a specified cutoff datetime.

    Examples:
        >>> from datetime import datetime
        >>> time_since = TimeSince()
        >>> times = [datetime(2019, 3, 1, 0, 0, 0, 1),
        ...          datetime(2019, 3, 1, 0, 0, 1, 0),
        ...          datetime(2019, 3, 1, 0, 2, 0, 0)]
        >>> cutoff_time = datetime(2019, 3, 1, 0, 0, 0, 0)
        >>> values = time_since(array=times, time=cutoff_time)
        >>> list(map(int, values))
        [-1000, -1000000000, -120000000000]
    """
    name = 'time_since'
    input_types = [[DatetimeTimeIndex], [Datetime]]
    return_type = Timedelta
    uses_calc_time = True

    def get_function(self):
        def pd_time_since(array, time):
            return (time - pd.DatetimeIndex(array)).values
        return pd_time_since


class DaysSince(TransformPrimitive):
    """Calculates the number of days from a value to a specified datetime.

    Examples:
        >>> from datetime import datetime
        >>> days_since = DaysSince()
        >>> dates = [datetime(2019, 3, 2, 0),
        ...          datetime(2019, 3, 10, 12),
        ...          datetime(2019, 3, 1, 0)]
        >>> cutoff_date = datetime(2019, 3, 1, 0, 0, 0, 0)
        >>> days_since(array=dates, time=cutoff_date).tolist()
        [-1, -10, 0]
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
    """Determines whether a value is present in a provided list.

    Examples:
        >>> items = ['string', 10.3, False]
        >>> is_in = IsIn(list_of_outputs=items)
        >>> is_in(['string', 10.5, False]).tolist()
        [True, False, True]
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
    """Compute the difference between the value in a list and the
    previous value.

    Description:
        Given a list of values and a corresponding list of item ID values,
        compute the difference from the previous occurrence of the item in
        the list. If an item is present only once, the result for this item
        will be `NaN`. Similarly, the result for the first occurrence of an
        item will always be `NaN`. If the values are datetimes, the output
        will be a timedelta.

    Examples:
        >>> diff = Diff()
        >>> values = [1, 10, 3, 4, 15]
        >>> labels = ['A', 'A', 'B', 'A', 'B']
        >>> diff(values, labels).tolist()
        [nan, 9.0, nan, -6.0, 12.0]

        If values are datetimes, difference will be a timedelta

        >>> from datetime import datetime
        >>> diff = Diff()
        >>> values = [datetime(2019, 3, 1, 0, 0, 0),
        ...          datetime(2019, 3, 1, 0, 1, 0),
        ...          datetime(2019, 3, 2, 0, 0, 0),
        ...          datetime(2019, 3, 1, 0, 1, 30)]
        >>> labels = ['A', 'A', 'B', 'A']
        >>> diff(values, labels).tolist()
        [NaT, Timedelta('0 days 00:01:00'), NaT, Timedelta('0 days 00:00:30')]
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
    """Negates a numeric value.

    Examples:
        >>> negate = Negate()
        >>> negate([1.0, 23.2, -7.0]).tolist()
        [-1.0, -23.2, 7.0]
    """
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
    """Negates a boolean value.

    Examples:
        >>> not_func = Not()
        >>> not_func([True, True, False]).tolist()
        [False, False, True]
    """
    name = "not"
    input_types = [Boolean]
    return_type = Boolean

    def generate_name(self, base_feature_names):
        return u"NOT({})".format(base_feature_names[0])

    def get_function(self):
        return np.logical_not


class Percentile(TransformPrimitive):
    """Determines the percentile rank for each value in a list.

    Examples:
        >>> percentile = Percentile()
        >>> percentile([10, 15, 1, 20]).tolist()
        [0.5, 0.75, 0.25, 1.0]

        Nan values are ignored when determining rank

        >>> percentile([10, 15, 1, None, 20]).tolist()
        [0.5, 0.75, 0.25, nan, 1.0]
    """
    name = 'percentile'
    uses_full_entity = True
    input_types = [Numeric]
    return_type = Numeric

    def get_function(self):
        return lambda array: pd.Series(array).rank(pct=True)


class Latitude(TransformPrimitive):
    """Returns the first tuple value in a list of LatLong tuples.
       For use with the LatLong variable type.

    Examples:
        >>> latitude = Latitude()
        >>> latitude([(42.4, -71.1),
        ...            (40.0, -122.4),
        ...            (41.2, -96.75)]).tolist()
        [42.4, 40.0, 41.2]
    """
    name = 'latitude'
    input_types = [LatLong]
    return_type = Numeric

    def get_function(self):
        return lambda array: pd.Series([x[0] for x in array])


class Longitude(TransformPrimitive):
    """Returns the second tuple value in a list of LatLong tuples.
       For use with the LatLong variable type.

    Examples:
        >>> longitude = Longitude()
        >>> longitude([(42.4, -71.1),
        ...            (40.0, -122.4),
        ...            (41.2, -96.75)]).tolist()
        [-71.1, -122.4, -96.75]
    """
    name = 'longitude'
    input_types = [LatLong]
    return_type = Numeric

    def get_function(self):
        return lambda array: pd.Series([x[1] for x in array])


class Haversine(TransformPrimitive):
    """Calculates the approximate haversine distance between two LatLong
        variable types.

        Args:
            unit (str): Determines the unit value to output. Could
                be `miles` or `kilometers`. Default is `miles`.

        Examples:
            >>> haversine = Haversine()
            >>> distances = haversine([(42.4, -71.1), (40.0, -122.4)],
            ...                       [(40.0, -122.4), (41.2, -96.75)])
            >>> np.round(distances, 3).tolist()
            [2631.231, 1343.289]

            Output units can be specified

            >>> haversine_km = Haversine(unit='kilometers')
            >>> distances_km = haversine_km([(42.4, -71.1), (40.0, -122.4)],
            ...                             [(40.0, -122.4), (41.2, -96.75)])
            >>> np.round(distances_km, 3).tolist()
            [4234.555, 2161.814]
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
