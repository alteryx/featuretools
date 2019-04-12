from __future__ import division

from builtins import str
from datetime import datetime

import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta

from featuretools.exceptions import NotEnoughData
from featuretools.utils import is_string


class Timedelta(object):
    """Represents differences in time.

    Timedeltas can be defined in multiple units. Supported units:

    - "ms" : milliseconds
    - "s" : seconds
    - "h" : hours
    - "m" : minutes
    - "d" : days
    - "o"/"observations" : number of individual events
    - "u"/"unit" : whatever unit associated column/number is

    Timedeltas can also be defined in terms of observations. In this case, the
    Timedelta represents the period spanned by `value` consecutive instances of
    the `entity`.

    For observation timedeltas, a

    >>> Timedelta(10, "s").value_in_seconds # 10 seconds
    10.0
    >>> three_observations_log = Timedelta(3, "observations", entity='log')
    >>> three_observations_log.get_name()
    '3 Observations'
    """

    _Observations = "o"

    _generic_unit = "u"
    _generic_expanded_unit = "units"

    # units for absolute times
    _time_units = ['ms', 's', 'h', 'm', 'd']

    _readable_units = {
        "ms": "Milliseconds",
        "s": "Seconds",
        "h": "Hours",
        "m": "Minutes",
        "d": "Days",
        "o": "Observations",
        "w": "Weeks",
        "Y": "Years",
        'u': 'Units'
    }

    _convert_to_days = {
        "w": 7
    }

    _readable_to_unit = {v.lower(): k for k, v in _readable_units.items()}

    def __init__(self, value, unit=None, entity=None, data=None, inclusive=False):
        """
        Args:
            value (float, str) : Value of timedelta, or string providing
                both unit and value.
            unit (str) : Unit of time delta.
            entity (str, optional) : Entity id to use if unit equals
                "observations".
            data (pd.Series, optional) : series of timestamps to use
                with observations. Can be calculated later.
            inclusive (bool, optional) : if True, include events that are
                exactly timedelta distance away from the original time/observation
        """
        # TODO: check if value is int or float
        if is_string(value):
            from featuretools.utils.wrangle import _check_timedelta
            td = _check_timedelta(value)
            value, unit = td.value, td.unit

        self.value = value
        self._original_unit = None  # to alert get_name that although we converted the unit to 'd' it was initially
        unit = self._check_unit_plural(unit)
        assert unit in self._readable_units or unit in self._readable_to_unit
        if unit in self._readable_to_unit:
            unit = self._readable_to_unit[unit]

        # weeks
        if unit in self._convert_to_days:
            self._original_unit = unit
            self.value = self.value * self._convert_to_days[unit]
            unit = 'd'

        self.unit = unit

        if unit == self._Observations and entity is None:
            raise Exception("Must define entity to use %s as unit" % (unit))

        self.entity = entity
        self.data = data

        self.inclusive = inclusive

    @classmethod
    def make_singular(cls, s):
        if len(s) > 1 and s.endswith('s'):
            return s[:-1]
        return s

    @classmethod
    def _check_unit_plural(cls, s):
        if len(s) > 1 and not s.endswith('s'):
            return (s + 's').lower()
        elif len(s) > 1:
            return s.lower()
        return s

    def get_name(self):
        if self.unit == self._generic_unit:
            return str(self.value)
        else:
            unit = self.readable_unit
            if self.readable_unit == "Weeks":
                # divide to convert back
                return "{} {}".format(self.value / self._convert_to_days["w"], unit)
            if self.value == 1:
                unit = self.make_singular(unit)

            return "{} {}".format(self.value, unit)

    def __eq__(self, other):
        if not isinstance(other, Timedelta):
            return False

        return (self.value == other.value and
                self.unit == other.unit and
                self.entity == other.entity and
                self.inclusive == other.inclusive)

    @property
    def readable_unit(self):
        if self._original_unit is not None:
            return self._readable_units[self._original_unit]
        return self._readable_units[self.unit]

    def get_pandas_timedelta(self):
        if self.is_absolute() and self.unit != self._generic_unit:
            return pd.Timedelta(self.value, self.unit)
        else:
            return None

    def view(self, unit):
        if self.is_absolute() and self.unit != self._generic_unit:
            return self.get_pandas_timedelta().view(unit)
        else:
            return None

    @property
    def value_in_seconds(self):
        if self.is_absolute() and self.unit != self._generic_unit:
            pd_td = self.get_pandas_timedelta()
            return pd_td.total_seconds()
        else:
            return None

    def is_absolute(self):
        return self.unit != self._Observations

    def __neg__(self):
        """Negate the timedelta"""
        return Timedelta(-self.value, self.unit, self.entity, self.data)

    def __call__(self, parent_entity, instance_id, entityset, inclusive=False):
        """
        Args:
            parent_entity (str) : Id of parent entity, from which our entity
                will be filtered.
            instance_id (str, int) : Instance ID on the parent entity used to
                select ids on this entity.
            entityset (BaseEntitySet) : Associated entityset from which to access data.
            inclusive (bool, optional]) : If True, include events that are
                exactly timedelta distance away from the original time/observation.

        Returns:
            :class:`Timedelta`
        """
        # this only does anything if our unit is 'observations.'
        if self.unit != self._Observations:
            return self

        time_index = entityset.entity_dict[self.entity].time_index
        data = entityset.related_instances(parent_entity, self.entity,
                                           [instance_id])[time_index]
        self.inclusive = inclusive

        # return copy with this info set
        return Timedelta(self.value, self.unit, self.entity, data,
                         inclusive=inclusive)

    def __radd__(self, time):
        """Add the Timedelta to a timestamp value"""
        if self.value > 0:
            return self._do_add(time, self.value)
        elif self.value < 0:
            return self._do_sub(time, -self.value)
        return time

    def __rsub__(self, time):
        """Subtract the Timedelta from a timestamp value"""
        if self.value > 0:
            return self._do_sub(time, self.value)
        elif self.value < 0:
            return self._do_add(time, -self.value)
        return time

    def _do_sub(self, time, value):
        assert value > 0, "Value must be greater than 0"

        if (self.unit == self._generic_unit and
                not isinstance(time, (pd.Timestamp, datetime))):
            return time - value
        elif self.unit != self._Observations:
            return add_td(time, -1 * value, self.unit)

        assert self.data is not None, "Must call timedelta with data"

        # reverse because we want to count backwards when we subtract
        if self.inclusive:
            all_times = self.data[self.data <= time].tolist()[::-1]
        else:
            all_times = self.data[self.data < time].tolist()[::-1]
        if self.unit == self._Observations:
            if len(all_times) < value:
                raise NotEnoughData()
            return all_times[value - 1]

        raise Exception("Invalid unit")

    def _do_add(self, time, value):
        assert value > 0, "Value must be greater than 0"

        if (self.unit == self._generic_unit and
                not isinstance(time, (pd.Timestamp, datetime))):
            return time + value
        elif self.unit not in [self._Observations]:
            return add_td(time, value, self.unit)

        assert self.data is not None, "Must call timedelta with data"

        all_times = self.data[self.data > time].tolist()
        if self.unit == self._Observations:
            if len(all_times) < value:
                raise NotEnoughData()

            return all_times[value - 1]

        raise Exception("Invalid unit")


def add_td(time, value, unit):
    if unit in ["ms", "s", "h", "m", "d"]:
        return time + pd.Timedelta(value, unit)
    elif unit == 'Y':
        if hasattr(time, '__len__'):
            is_array = isinstance(time, np.ndarray)
            new_time = pd.Series(time).apply(lambda x: x + relativedelta(years=value))
            if is_array:
                new_time = new_time.values
            return new_time
        else:
            return time + relativedelta(years=value)
    else:
        raise ValueError("Invalid Unit")
