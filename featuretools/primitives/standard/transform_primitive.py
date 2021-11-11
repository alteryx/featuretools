import warnings

import numpy as np
import pandas as pd
from woodwork.column_schema import ColumnSchema
from woodwork.logical_types import (
    URL,
    AgeFractional,
    Boolean,
    BooleanNullable,
    Categorical,
    Datetime,
    EmailAddress,
    LatLong,
    NaturalLanguage,
    Ordinal
)

from featuretools.primitives.base.transform_primitive_base import (
    TransformPrimitive
)
from featuretools.utils import convert_time_units
from featuretools.utils.common_tld_utils import COMMON_TLDS
from featuretools.utils.gen_utils import Library


class IsNull(TransformPrimitive):
    """Determines if a value is null.

    Examples:
        >>> is_null = IsNull()
        >>> is_null([1, None, 3]).tolist()
        [False, True, False]
    """
    name = "is_null"
    input_types = [ColumnSchema()]
    return_type = ColumnSchema(logical_type=Boolean)
    compatibility = [Library.PANDAS, Library.DASK, Library.KOALAS]
    description_template = "whether {} is null"

    def get_function(self):
        def isnull(array):
            return array.isnull()
        return isnull


class Absolute(TransformPrimitive):
    """Computes the absolute value of a number.

    Examples:
        >>> absolute = Absolute()
        >>> absolute([3.0, -5.0, -2.4]).tolist()
        [3.0, 5.0, 2.4]
    """
    name = "absolute"
    input_types = [ColumnSchema(semantic_tags={'numeric'})]
    return_type = ColumnSchema(semantic_tags={'numeric'})
    compatibility = [Library.PANDAS, Library.DASK, Library.KOALAS]
    description_template = "the absolute value of {}"

    def get_function(self):
        return np.absolute


class TimeSincePrevious(TransformPrimitive):
    """Compute the time since the previous entry in a list.

    Args:
        unit (str): Defines the unit of time to count from.
            Defaults to Seconds. Acceptable values:
            years, months, days, hours, minutes, seconds, milliseconds, nanoseconds

    Description:
        Given a list of datetimes, compute the time in seconds elapsed since
        the previous item in the list. The result for the first item in the
        list will always be `NaN`.

    Examples:
        >>> from datetime import datetime
        >>> time_since_previous = TimeSincePrevious()
        >>> dates = [datetime(2019, 3, 1, 0, 0, 0),
        ...          datetime(2019, 3, 1, 0, 2, 0),
        ...          datetime(2019, 3, 1, 0, 3, 0),
        ...          datetime(2019, 3, 1, 0, 2, 30),
        ...          datetime(2019, 3, 1, 0, 10, 0)]
        >>> time_since_previous(dates).tolist()
        [nan, 120.0, 60.0, -30.0, 450.0]
    """
    name = "time_since_previous"
    input_types = [ColumnSchema(logical_type=Datetime, semantic_tags={'time_index'})]
    return_type = ColumnSchema(semantic_tags={'numeric'})
    description_template = "the time since the previous instance of {}"

    def __init__(self, unit="seconds"):
        self.unit = unit.lower()

    def get_function(self):
        def pd_diff(values):
            return convert_time_units(values.diff().apply(lambda x: x.total_seconds()), self.unit)
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
    input_types = [ColumnSchema(logical_type=Datetime)]
    return_type = ColumnSchema(logical_type=Ordinal(order=list(range(1, 32))), semantic_tags={'category'})
    compatibility = [Library.PANDAS, Library.DASK, Library.KOALAS]
    description_template = "the day of the month of {}"

    def get_function(self):
        def day(vals):
            return vals.dt.day
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
    input_types = [ColumnSchema(logical_type=Datetime)]
    return_type = ColumnSchema(logical_type=Ordinal(order=list(range(24))), semantic_tags={'category'})
    compatibility = [Library.PANDAS, Library.DASK, Library.KOALAS]
    description_template = 'the hour value of {}'

    def get_function(self):
        def hour(vals):
            return vals.dt.hour
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
    input_types = [ColumnSchema(logical_type=Datetime)]
    return_type = ColumnSchema(logical_type=Ordinal(order=list(range(60))), semantic_tags={'category'})
    compatibility = [Library.PANDAS, Library.DASK, Library.KOALAS]
    description_template = "the seconds value of {}"

    def get_function(self):
        def second(vals):
            return vals.dt.second
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
    input_types = [ColumnSchema(logical_type=Datetime)]
    return_type = ColumnSchema(logical_type=Ordinal(order=list(range(60))), semantic_tags={'category'})
    compatibility = [Library.PANDAS, Library.DASK, Library.KOALAS]
    description_template = "the minutes value of {}"

    def get_function(self):
        def minute(vals):
            return vals.dt.minute
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
    input_types = [ColumnSchema(logical_type=Datetime)]
    return_type = ColumnSchema(logical_type=Ordinal(order=list(range(1, 54))), semantic_tags={'category'})
    compatibility = [Library.PANDAS, Library.DASK, Library.KOALAS]
    description_template = "the week of the year of {}"

    def get_function(self):
        def week(vals):
            warnings.filterwarnings("ignore",
                                    message=("Series.dt.weekofyear and Series.dt.week "
                                             "have been deprecated."),
                                    module="featuretools"
                                    )
            return vals.dt.week
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
    input_types = [ColumnSchema(logical_type=Datetime)]
    return_type = ColumnSchema(logical_type=Ordinal(order=list(range(1, 13))), semantic_tags={'category'})
    compatibility = [Library.PANDAS, Library.DASK, Library.KOALAS]
    description_template = "the month of {}"

    def get_function(self):
        def month(vals):
            return vals.dt.month
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
    input_types = [ColumnSchema(logical_type=Datetime)]
    return_type = ColumnSchema(logical_type=Ordinal(order=list(range(1, 3000))), semantic_tags={'category'})
    compatibility = [Library.PANDAS, Library.DASK, Library.KOALAS]
    description_template = "the year of {}"

    def get_function(self):
        def year(vals):
            return vals.dt.year
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
    input_types = [ColumnSchema(logical_type=Datetime)]
    return_type = ColumnSchema(logical_type=BooleanNullable)
    compatibility = [Library.PANDAS, Library.DASK, Library.KOALAS]
    description_template = "whether {} occurred on a weekend"

    def get_function(self):
        def is_weekend(vals):
            return vals.dt.weekday > 4
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
    input_types = [ColumnSchema(logical_type=Datetime)]
    return_type = ColumnSchema(logical_type=Ordinal(order=list(range(7))), semantic_tags={'category'})
    compatibility = [Library.PANDAS, Library.DASK, Library.KOALAS]
    description_template = "the day of the week of {}"

    def get_function(self):
        def weekday(vals):
            return vals.dt.weekday
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
    input_types = [ColumnSchema(logical_type=NaturalLanguage)]
    return_type = ColumnSchema(semantic_tags={'numeric'})
    compatibility = [Library.PANDAS, Library.DASK, Library.KOALAS]
    description_template = "the number of characters in {}"

    def get_function(self):
        def character_counter(array):
            return array.fillna('').str.len()
        return character_counter


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
    input_types = [ColumnSchema(logical_type=NaturalLanguage)]
    return_type = ColumnSchema(semantic_tags={'numeric'})
    compatibility = [Library.PANDAS, Library.DASK, Library.KOALAS]
    description_template = "the number of words in {}"

    def get_function(self):
        def word_counter(array):
            return array.fillna('').str.count(' ') + 1
        return word_counter


class TimeSince(TransformPrimitive):
    """Calculates time from a value to a specified cutoff datetime.

    Args:
        unit (str): Defines the unit of time to count from.
            Defaults to Seconds. Acceptable values:
            years, months, days, hours, minutes, seconds, milliseconds, nanoseconds

    Examples:
        >>> from datetime import datetime
        >>> time_since = TimeSince()
        >>> times = [datetime(2019, 3, 1, 0, 0, 0, 1),
        ...          datetime(2019, 3, 1, 0, 0, 1, 0),
        ...          datetime(2019, 3, 1, 0, 2, 0, 0)]
        >>> cutoff_time = datetime(2019, 3, 1, 0, 0, 0, 0)
        >>> values = time_since(times, time=cutoff_time)
        >>> list(map(int, values))
        [0, -1, -120]

        Change output to nanoseconds

        >>> from datetime import datetime
        >>> time_since_nano = TimeSince(unit='nanoseconds')
        >>> times = [datetime(2019, 3, 1, 0, 0, 0, 1),
        ...          datetime(2019, 3, 1, 0, 0, 1, 0),
        ...          datetime(2019, 3, 1, 0, 2, 0, 0)]
        >>> cutoff_time = datetime(2019, 3, 1, 0, 0, 0, 0)
        >>> values = time_since_nano(times, time=cutoff_time)
        >>> list(map(lambda x: int(round(x)), values))
        [-1000, -1000000000, -120000000000]
    """
    name = 'time_since'
    input_types = [ColumnSchema(logical_type=Datetime)]
    return_type = ColumnSchema(semantic_tags={'numeric'})
    uses_calc_time = True
    compatibility = [Library.PANDAS, Library.DASK]
    description_template = "the time from {} to the cutoff time"

    def __init__(self, unit="seconds"):
        self.unit = unit.lower()

    def get_function(self):
        def pd_time_since(array, time):
            return convert_time_units((time - array).dt.total_seconds(), self.unit)
        return pd_time_since


class IsIn(TransformPrimitive):
    """Determines whether a value is present in a provided list.

    Examples:
        >>> items = ['string', 10.3, False]
        >>> is_in = IsIn(list_of_outputs=items)
        >>> is_in(['string', 10.5, False]).tolist()
        [True, False, True]
    """
    name = "isin"
    input_types = [ColumnSchema()]
    return_type = ColumnSchema(logical_type=Boolean)
    compatibility = [Library.PANDAS, Library.DASK, Library.KOALAS]

    def __init__(self, list_of_outputs=None):
        self.list_of_outputs = list_of_outputs
        if not list_of_outputs:
            stringified_output_list = '[]'
        else:
            stringified_output_list = ', '.join([str(x) for x in list_of_outputs])
        self.description_template = "whether {{}} is in {}".format(stringified_output_list)

    def get_function(self):
        def pd_is_in(array):
            return array.isin(self.list_of_outputs or [])
        return pd_is_in

    def generate_name(self, base_feature_names):
        return u"%s.isin(%s)" % (base_feature_names[0],
                                 str(self.list_of_outputs))


class Diff(TransformPrimitive):
    """Compute the difference between the value in a list and the
    previous value in that list.

    Description:
        Given a list of values, compute the difference from the previous
        item in the list. The result for the first element of the list will
        always be `NaN`. If the values are datetimes, the output will be a
        timedelta.

    Examples:
        >>> diff = Diff()
        >>> values = [1, 10, 3, 4, 15]
        >>> diff(values).tolist()
        [nan, 9.0, -7.0, 1.0, 11.0]
    """
    name = "diff"
    input_types = [ColumnSchema(semantic_tags={'numeric'})]
    return_type = ColumnSchema(semantic_tags={'numeric'})
    uses_full_dataframe = True
    description_template = "the difference from the previous value of {}"

    def get_function(self):
        def pd_diff(values):
            return values.diff()
        return pd_diff


class Negate(TransformPrimitive):
    """Negates a numeric value.

    Examples:
        >>> negate = Negate()
        >>> negate([1.0, 23.2, -7.0]).tolist()
        [-1.0, -23.2, 7.0]
    """
    name = "negate"
    input_types = [ColumnSchema(semantic_tags={'numeric'})]
    return_type = ColumnSchema(semantic_tags={'numeric'})
    compatibility = [Library.PANDAS, Library.DASK, Library.KOALAS]
    description_template = "the negation of {}"

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
    input_types = [[ColumnSchema(logical_type=Boolean)], [ColumnSchema(logical_type=BooleanNullable)]]
    return_type = ColumnSchema(logical_type=BooleanNullable)
    compatibility = [Library.PANDAS, Library.DASK, Library.KOALAS]
    description_template = "the negation of {}"

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
    uses_full_dataframe = True
    input_types = [ColumnSchema(semantic_tags={'numeric'})]
    return_type = ColumnSchema(semantic_tags={'numeric'})
    description_template = "the percentile rank of {}"

    def get_function(self):
        return lambda array: array.rank(pct=True)


class Latitude(TransformPrimitive):
    """Returns the first tuple value in a list of LatLong tuples.
       For use with the LatLong logical type.

    Examples:
        >>> latitude = Latitude()
        >>> latitude([(42.4, -71.1),
        ...            (40.0, -122.4),
        ...            (41.2, -96.75)]).tolist()
        [42.4, 40.0, 41.2]
    """
    name = 'latitude'
    input_types = [ColumnSchema(logical_type=LatLong)]
    return_type = ColumnSchema(semantic_tags={'numeric'})
    description_template = "the latitude of {}"

    def get_function(self):
        def latitude(latlong):
            return latlong.map(lambda x: x[0] if isinstance(x, tuple) else np.nan)
        return latitude


class Longitude(TransformPrimitive):
    """Returns the second tuple value in a list of LatLong tuples.
       For use with the LatLong logical type.

    Examples:
        >>> longitude = Longitude()
        >>> longitude([(42.4, -71.1),
        ...            (40.0, -122.4),
        ...            (41.2, -96.75)]).tolist()
        [-71.1, -122.4, -96.75]
    """
    name = 'longitude'
    input_types = [ColumnSchema(logical_type=LatLong)]
    return_type = ColumnSchema(semantic_tags={'numeric'})
    description_template = "the longitude of {}"

    def get_function(self):
        def longitude(latlong):
            return latlong.map(lambda x: x[1] if isinstance(x, tuple) else np.nan)
        return longitude


class Haversine(TransformPrimitive):
    """Calculates the approximate haversine distance between two LatLong columns.

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
    input_types = [ColumnSchema(logical_type=LatLong), ColumnSchema(logical_type=LatLong)]
    return_type = ColumnSchema(semantic_tags={'numeric'})
    commutative = True

    def __init__(self, unit='miles'):
        valid_units = ['miles', 'kilometers']
        if unit not in valid_units:
            error_message = 'Invalid unit %s provided. Must be one of %s' % (unit, valid_units)
            raise ValueError(error_message)
        self.unit = unit
        self.description_template = "the haversine distance in {} between {{}} and {{}}".format(self.unit)

    def get_function(self):
        def haversine(latlong1, latlong2):
            lat_1s = np.array([x[0] if isinstance(x, tuple) else np.nan for x in latlong1])
            lon_1s = np.array([x[1] if isinstance(x, tuple) else np.nan for x in latlong1])
            lat_2s = np.array([x[0] if isinstance(x, tuple) else np.nan for x in latlong2])
            lon_2s = np.array([x[1] if isinstance(x, tuple) else np.nan for x in latlong2])
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


class Age(TransformPrimitive):
    """Calculates the age in years as a floating point number given a
       date of birth.

    Description:
        Age in years is computed by calculating the number of days between
        the date of birth and the reference time and dividing the result
        by 365.

    Examples:
        Determine the age of three people as of Jan 1, 2019
        >>> import pandas as pd
        >>> reference_date = pd.to_datetime("01-01-2019")
        >>> age = Age()
        >>> input_ages = [pd.to_datetime("01-01-2000"),
        ...               pd.to_datetime("05-30-1983"),
        ...               pd.to_datetime("10-17-1997")]
        >>> age(input_ages, time=reference_date).tolist()
        [19.013698630136986, 35.61643835616438, 21.221917808219178]
    """
    name = "age"
    input_types = [ColumnSchema(logical_type=Datetime, semantic_tags={'date_of_birth'})]
    return_type = ColumnSchema(logical_type=AgeFractional, semantic_tags={'numeric'})
    uses_calc_time = True
    compatibility = [Library.PANDAS, Library.DASK]
    description_template = "the age from {}"

    def get_function(self):
        def age(x, time=None):
            return (time - x).dt.days / 365
        return age


class URLToDomain(TransformPrimitive):
    """Determines the domain of a url.

    Description:
        Calculates the label to identify the network domain of a URL. Supports
        urls with or without protocol as well as international country domains.

    Examples:
        >>> url_to_domain = URLToDomain()
        >>> urls =  ['https://play.google.com',
        ...          'http://www.google.co.in',
        ...          'www.facebook.com']
        >>> url_to_domain(urls).tolist()
        ['play.google.com', 'google.co.in', 'facebook.com']
    """
    name = "url_to_domain"
    input_types = [ColumnSchema(logical_type=URL)]
    return_type = ColumnSchema(logical_type=Categorical, semantic_tags={'category'})

    def get_function(self):
        def url_to_domain(x):
            p = r'^(?:https?:\/\/)?(?:[^@\/\n]+@)?(?:www\.)?([^:\/?\n]+)'
            return x.str.extract(p, expand=False)
        return url_to_domain


class URLToProtocol(TransformPrimitive):
    """Determines the protocol (http or https) of a url.

    Description:
        Extract the protocol of a url using regex.
        It will be either https or http. Returns nan if
        the url doesn't contain a protocol.

    Examples:
        >>> url_to_protocol = URLToProtocol()
        >>> urls =  ['https://play.google.com',
        ...          'http://www.google.co.in',
        ...          'www.facebook.com']
        >>> url_to_protocol(urls).to_list()
        ['https', 'http', nan]
    """
    name = "url_to_protocol"
    input_types = [ColumnSchema(logical_type=URL)]
    return_type = ColumnSchema(logical_type=Categorical, semantic_tags={'category'})

    def get_function(self):
        def url_to_protocol(x):
            p = r'^(https|http)(?:\:)'
            return x.str.extract(p, expand=False)
        return url_to_protocol


class URLToTLD(TransformPrimitive):
    """Determines the top level domain of a url.

    Description:
        Extract the top level domain of a url, using regex,
        and a list of common top level domains. Returns nan if
        the url is invalid or null.
        Common top level domains were pulled from this list:
        https://www.hayksaakian.com/most-popular-tlds/

    Examples:
        >>> url_to_tld = URLToTLD()
        >>> urls = ['https://www.google.com', 'http://www.google.co.in',
        ...         'www.facebook.com']
        >>> url_to_tld(urls).to_list()
        ['com', 'in', 'com']
    """
    name = "url_to_tld"
    input_types = [ColumnSchema(logical_type=URL)]
    return_type = ColumnSchema(logical_type=Categorical, semantic_tags={'category'})

    def get_function(self):
        self.tlds_pattern = r'(?:\.({}))'.format('|'.join(COMMON_TLDS))

        def url_to_domain(x):
            p = r'^(?:https?:\/\/)?(?:[^@\/\n]+@)?(?:www\.)?([^:\/?\n]+)'
            return x.str.extract(p, expand=False)

        def url_to_tld(x):
            domains = url_to_domain(x)
            df = domains.str.extractall(self.tlds_pattern)
            matches = df.groupby(level=0).last()[0]
            return matches.reindex(x.index)
        return url_to_tld


class IsFreeEmailDomain(TransformPrimitive):
    """Determines if an email address is from a free email domain.

    Description:
        EmailAddress input should be a string. Will return Nan
        if an invalid email address is provided, or if the input is
        not a string. The list of free email domains used in this primitive
        was obtained from https://github.com/willwhite/freemail/blob/master/data/free.txt.

    Examples:
        >>> is_free_email_domain = IsFreeEmailDomain()
        >>> is_free_email_domain(['name@gmail.com', 'name@featuretools.com']).tolist()
        [True, False]
    """
    name = "is_free_email_domain"
    input_types = [ColumnSchema(logical_type=EmailAddress)]
    return_type = ColumnSchema(logical_type=BooleanNullable)

    filename = "free_email_provider_domains.txt"

    def get_function(self):
        file_path = self.get_filepath(self.filename)

        free_domains = pd.read_csv(file_path, header=None, names=['domain'])
        free_domains['domain'] = free_domains.domain.str.strip()

        def is_free_email_domain(emails):
            # if the input is empty return an empty Series
            if len(emails) == 0:
                return pd.Series([])

            emails_df = pd.DataFrame({'email': emails})

            # if all emails are NaN expand won't propogate NaNs and will fail on indexing
            if emails_df['email'].isnull().all():
                emails_df['domain'] = np.nan
            else:
                # .str.strip() and .str.split() return NaN for NaN values and propogate NaNs into new columns
                emails_df['domain'] = emails_df['email'].str.strip().str.split('@', expand=True)[1]

            emails_df['is_free'] = emails_df['domain'].isin(free_domains['domain'])

            # if there are any NaN domain values, change the series type to allow for
            # both bools and NaN values and set is_free to NaN for the NaN domains
            if emails_df['domain'].isnull().values.any():
                emails_df['is_free'] = emails_df['is_free'].astype(np.object)
                emails_df.loc[emails_df['domain'].isnull(), 'is_free'] = np.nan
            return emails_df.is_free.values
        return is_free_email_domain


class EmailAddressToDomain(TransformPrimitive):
    """Determines the domain of an email

    Description:
        EmailAddress input should be a string. Will return Nan
        if an invalid email address is provided, or if the input is
        not a string.

    Examples:
        >>> email_address_to_domain = EmailAddressToDomain()
        >>> email_address_to_domain(['name@gmail.com', 'name@featuretools.com']).tolist()
        ['gmail.com', 'featuretools.com']
    """
    name = "email_address_to_domain"
    input_types = [ColumnSchema(logical_type=EmailAddress)]
    return_type = ColumnSchema(logical_type=Categorical, semantic_tags={'category'})

    def get_function(self):
        def email_address_to_domain(emails):
            # if the input is empty return an empty Series
            if len(emails) == 0:
                return pd.Series([])

            emails_df = pd.DataFrame({'email': emails})

            # if all emails are NaN expand won't propogate NaNs and will fail on indexing
            if emails_df['email'].isnull().all():
                emails_df['domain'] = np.nan
                emails_df['domain'] = emails_df['domain'].astype(object)
            else:
                # .str.strip() and .str.split() return NaN for NaN values and propogate NaNs into new columns
                emails_df['domain'] = emails_df['email'].str.strip().str.split('@', expand=True)[1]
            return emails_df.domain.values
        return email_address_to_domain
