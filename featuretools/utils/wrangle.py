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
    When using generic units, can drop the unit
    1
    2
    '1'
    '2'
    When using observations, need to provide an entity as either a tuple or a separate arg
    ('2o', 'logs')
    ('2 o', 'logs')
    ('2 Observations', 'logs')
    ('2 observations', 'logs')
    ('2 observation', 'logs')
    If an entity is provided and no unit is provided, assume observations (instead of generic units)
    (2, 'logs')
    ('2', 'logs')



    """
    if td is None:
        return td
    if isinstance(td, Timedelta):
        if td.entity is not None and entity_id is not None and td.entity != entity_id:
            raise ValueError("Timedelta entity {} different from passed entity {}".format(td.entity, entity_id))
        if td.entity is not None and related_entity_id is not None and td.entity == related_entity_id:
            raise ValueError("Timedelta entity {} same as passed related entity {}".format(td.entity, related_entity_id))
        return td
    elif not (is_string(td) or isinstance(td, (tuple, int, float))):
        raise ValueError("Unable to parse timedelta: {}".format(td))

    # TODO: allow observations from an entity in string

    if isinstance(td, tuple):
        if entity_id is None:
            entity_id = td[1]
        td = td[0]

    value = None
    try:
        value = int(td)
    except Exception:
        try:
            value = float(td)
        except Exception:
            pass
    if value is not None and entity_id is not None:
        unit = 'o'
    elif value is not None:
        unit = 'u'
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
    return Timedelta(value, unit, entity=entity_id)


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
    if df1.empty and not df2.empty:
        return False
    elif not df1.empty and df2.empty:
        return False
    elif not df1.empty and not df2.empty:
        for df in [df1, df2]:
            obj = df.select_dtypes('object').columns
            df[obj] = df[obj].astype('unicode')
        for c in df1:
            normal_compare = True
            if df1[c].dtype == object:
                dropped = df1[c].dropna()
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
                result = df1[c] == df2[c]
                result[pd.isnull(df1[c]) == pd.isnull(df2)[c]] = True
                if not result.all():
                    return False
    return True
