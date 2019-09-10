import re
from datetime import datetime

import numpy as np
import pandas as pd

from featuretools import variable_types
from featuretools.entityset.timedelta import Timedelta
from featuretools.utils import is_string


def _check_timedelta(td, entity_id=None, related_entity_id=None):
    """
    Convert strings to Timedelta objects
    Allows for both shortform and longform units, as well as any form of capitalization
    '2 Minutes'
    '2 minutes'
    '2 m'
    '1 Minute'
    '1 minute'
    '1 m'
    '1 units'
    '1 Units'
    '1 u'
    Shortform is fine if space is dropped
    '2m'
    '1u"
    If a pd.Timedelta object is passed, units will be converted to seconds due to the underlying representation
        of pd .Timedelta.
    """
    if td is None:
        return td
    if isinstance(td, Timedelta):
        return td
    elif not (is_string(td) or isinstance(td, pd.Timedelta) or isinstance(td, (int, float))):
        raise ValueError("Unable to parse timedelta: {}".format(td))

    value = None
    try:
        value = int(td)
    except Exception:
        try:
            value = float(td)
        except Exception:
            pass
    if isinstance(td, pd.Timedelta):
        unit = 's'
        value = td.total_seconds()
    else:
        pattern = '([0-9]+) *([a-zA-Z]+)$'
        match = re.match(pattern, td)
        value, unit = match.groups()
        try:
            value = int(value)
        except Exception:
            try:
                value = float(value)
            except Exception:
                raise ValueError("Unable to parse value {} from ".format(value) +
                                 "timedelta string: {}".format(td))
    return Timedelta(value, unit)


def _check_time_against_column(time, time_column):
    '''
    Check to make sure that time is compatible with time_column,
    where time could be a timestamp, or a Timedelta, number, or None,
    and time_column is a Variable. Compatibility means that
    arithmetic can be performed between time and elements of time_columnj

    If time is None, then we don't care if arithmetic can be performed
    (presumably it won't ever be performed)
    '''
    if time is None:
        return True
    elif isinstance(time, (int, float)):
        return isinstance(time_column,
                          variable_types.Numeric)
    elif isinstance(time, (pd.Timestamp, datetime)):
        return isinstance(time_column,
                          variable_types.Datetime)
    elif isinstance(time, Timedelta):
        return (isinstance(time_column, (variable_types.Datetime, variable_types.DatetimeTimeIndex)) or
                (isinstance(time_column, (variable_types.Ordinal, variable_types.Numeric, variable_types.TimeIndex)) and
                 time.unit not in Timedelta._time_units))
    else:
        return False


def _check_time_type(time):
    '''
    Checks if `time` is an instance of common int, float, or datetime types.
    Returns "numeric", "datetime", or "unknown" based on results
    '''
    time_type = None
    if isinstance(time, (datetime, np.datetime64)):
        time_type = variable_types.DatetimeTimeIndex
    elif isinstance(time, (int, float)) or np.issubdtype(time, np.integer) or np.issubdtype(time, np.floating):
        time_type = variable_types.NumericTimeIndex
    return time_type


def _dataframes_equal(df1, df2):
    # ^ means XOR
    if df1.empty ^ df2.empty:
        return False
    elif not df1.empty and not df2.empty:
        if not set(df1.columns) == set(df2.columns):
            return False

        for c in df1:
            df1c = df1[c]
            df2c = df2[c]
            if df1c.dtype == object:
                df1c = df1c.astype('unicode')
            if df2c.dtype == object:
                df2c = df2c.astype('unicode')

            normal_compare = True
            if df1c.dtype == object:
                dropped = df1c.dropna()
                if not dropped.empty:
                    if isinstance(dropped.iloc[0], tuple):
                        dropped2 = df2[c].dropna()
                        normal_compare = False
                        for i in range(len(dropped.iloc[0])):
                            try:
                                equal = dropped.apply(lambda x: x[i]).equals(
                                    dropped2.apply(lambda x: x[i]))
                            except IndexError:
                                raise IndexError("If column data are tuples, they must all be the same length")
                            if not equal:
                                return False
            if normal_compare:
                # handle nan equality correctly
                # This way is much faster than df1.equals(df2)
                result = df1c == df2c
                result[pd.isnull(df1c) == pd.isnull(df2c)] = True
                if not result.all():
                    return False
    return True


def _is_s3(string):
    '''
    Checks if the given string is a s3 path.
    Returns a boolean.
    '''
    return "s3://" in string


def _is_url(string):
    '''
    Checks if the given string is an url path.
    Returns a boolean.
    '''
    return 'http' in string
