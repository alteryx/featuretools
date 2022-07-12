import re
import tarfile
from datetime import datetime

import numpy as np
import pandas as pd
from woodwork.logical_types import Datetime, Ordinal

from featuretools.entityset.timedelta import Timedelta


def _check_timedelta(td):
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
        of pd.Timedelta.
    If a pd.DateOffset object is passed, it will be converted to a Featuretools Timedelta if it has one
        temporal parameter. Otherwise, it will remain a pd.DateOffset.
    """
    if td is None:
        return td
    if isinstance(td, Timedelta):
        return td
    elif not isinstance(td, (int, float, str, pd.DateOffset, pd.Timedelta)):
        raise ValueError("Unable to parse timedelta: {}".format(td))
    if isinstance(td, pd.Timedelta):
        unit = "s"
        value = td.total_seconds()
        times = {unit: value}
        return Timedelta(times, delta_obj=td)
    elif isinstance(td, pd.DateOffset):
        # DateOffsets
        if td.__class__.__name__ != "DateOffset":
            if hasattr(td, "__dict__"):
                # Special offsets (such as BDay) - prior to pandas 1.0.0
                value = td.__dict__["n"]
            else:
                # Special offsets (such as BDay) - after pandas 1.0.0
                value = td.n
            unit = td.__class__.__name__
            times = dict([(unit, value)])
        else:
            times = dict()
            for td_unit, td_value in td.kwds.items():
                times[td_unit] = td_value
        return Timedelta(times, delta_obj=td)
    else:
        pattern = "([0-9]+) *([a-zA-Z]+)$"
        match = re.match(pattern, td)
        value, unit = match.groups()
        try:
            value = int(value)
        except Exception:
            try:
                value = float(value)
            except Exception:
                raise ValueError(
                    "Unable to parse value {} from ".format(value)
                    + "timedelta string: {}".format(td),
                )
        times = {unit: value}
        return Timedelta(times)


def _check_time_against_column(time, time_column):
    """
    Check to make sure that time is compatible with time_column,
    where time could be a timestamp, or a Timedelta, number, or None,
    and time_column is a Woodwork initialized column. Compatibility means that
    arithmetic can be performed between time and elements of time_column

    If time is None, then we don't care if arithmetic can be performed
    (presumably it won't ever be performed)
    """
    if time is None:
        return True
    elif isinstance(time, (int, float)):
        return time_column.ww.schema.is_numeric
    elif isinstance(time, (pd.Timestamp, datetime, pd.DateOffset)):
        return time_column.ww.schema.is_datetime
    elif isinstance(time, Timedelta):
        if time_column.ww.schema.is_datetime:
            return True
        elif time.unit not in Timedelta._time_units:
            if (
                isinstance(time_column.ww.logical_type, Ordinal)
                or "numeric" in time_column.ww.semantic_tags
                or "time_index" in time_column.ww.semantic_tags
            ):
                return True
    return False


def _check_time_type(time):
    """
    Checks if `time` is an instance of common int, float, or datetime types.
    Returns "numeric" or Datetime based on results
    """
    time_type = None
    if isinstance(time, (datetime, np.datetime64)):
        time_type = Datetime
    elif (
        isinstance(time, (int, float))
        or np.issubdtype(time, np.integer)
        or np.issubdtype(time, np.floating)
    ):
        time_type = "numeric"
    return time_type


def _is_s3(string):
    """
    Checks if the given string is a s3 path.
    Returns a boolean.
    """
    return string.startswith("s3://")


def _is_url(string):
    """
    Checks if the given string is an url path.
    Returns a boolean.
    """
    return string.startswith("http")


def _is_local_tar(string):
    """
    Checks if the given string is a local tarfile path.
    Returns a boolean.
    """
    return string.endswith(".tar") and tarfile.is_tarfile(string)
