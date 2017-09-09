import pandas as pd
from featuretools import variable_types
from featuretools.entityset.timedelta import Timedelta
from datetime import datetime
import re
import numpy as np


def flatten_2d(array):
    """
    Converts numpy array to a 2-dimensional matrix.
    If an array has any list-like elements, they will be expanded into their own columns

    Example:
    x = np.array([[np.array([1,2]), 2],
                  [np.array([3,4]), 4]])
    flattened = flatten_2d(x)
    flattened
    >>> np.array([[1, 2, 2],
                  [3, 4, 4]])
    """
    nonscalars = {c: len(v)
                  for c, v in enumerate(array[0, :])
                  if not np.isscalar(v)}
    new_width = array.shape[1] + sum([v - 1 for v in nonscalars.values()])
    new_matrix = np.zeros((array.shape[0], new_width))
    i = 0
    c = 0
    while i < new_width:
        if c in nonscalars:
            c_len = nonscalars[c]
            values = np.concatenate(array[:, c]).reshape(array.shape[0], c_len)
            new_matrix[:, i: i + c_len] = values
            i += c_len
        else:
            new_matrix[:, i] = array[:, c]
            i += 1
        c += 1
    return new_matrix


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
    elif not isinstance(td, (basestring, tuple, int, float)):
        raise ValueError("Unable to parse timedelta: {}".format(td))

    # TODO: allow observations from an entity in string

    if isinstance(td, tuple):
        if entity_id is None:
            entity_id = td[1]
        td = td[0]

    value = None
    try:
        value = int(td)
    except:
        try:
            value = float(td)
        except:
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
        except:
            try:
                value = float(value)
            except:
                raise ValueError("Unable to parse value {} from ".format(value) +
                                 "timedelta string: {}".format(td))
    return Timedelta(value, unit, entity=entity_id)


def _check_variable_list(variables, entity, ignore_unknown=False):
    """Ensures a list of of values representing variables is
        a list of variable instances"""
    if len(variables) == 0:
        return []

    if ignore_unknown:
        return [_v for _v in [_check_variable(v, entity, ignore_unknown=ignore_unknown) for v in variables]
                if _v]
    else:
        return [_check_variable(v, entity, ignore_unknown=False) for v in variables]

    raise Exception("Couldn't handle list of variables")


def _check_variable(variable, entity, ignore_unknown=False):
    """Ensures a value representing a variable is
        a variable instance"""
    if not isinstance(variable, variable_types.Variable):
        if ignore_unknown and variable not in entity.variables:
            return None
        else:
            return entity[variable]
    else:
        if ignore_unknown and variable.id not in entity:
            return None
        else:
            return variable


def _check_entity(entity, entityset):
    """Ensures a value representing an entity is
        an entity instance"""
    from featuretools.entityset.base_entity import BaseEntity
    if isinstance(entity, BaseEntity):
        return entity
    return entityset[entity]


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


def _check_time_against_time(time1, time2):
    '''
    Check to make sure that time1 is compatible with time2,
    where time1 could be a timestamp, timedelta, number, or None,
    and time2 could be a number or Timedelta. Compatibility means that
    arithmetic can be performed between time1 and time2.

    If time1 is None, then we don't care if arithmetic can be performed
    (presumably it won't ever be performed)
    '''
    if time1 is None:
        return True
    elif isinstance(time1, (int, float)):
        return isinstance(time2, (int, float))
    elif isinstance(time1, Timedelta):
        if isinstance(time2, (pd.Timestamp, datetime)):
            return True
        elif time1.unit == Timedelta._Observations:
            return True
        elif time1.unit == Timedelta._generic_unit:
            return True
        else:
            return False
    elif isinstance(time1, (pd.Timestamp, datetime)):
        return isinstance(time2, (pd.Timedelta))
    else:
        return False
