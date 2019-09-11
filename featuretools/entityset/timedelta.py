from __future__ import division

import pandas as pd
from dateutil.relativedelta import relativedelta

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
    - "mo" : months
    - "Y" : years

    Timedeltas can also be defined in terms of observations. In this case, the
    Timedelta represents the period spanned by `value`.

    For observation timedeltas:
    >>> three_observations_log = Timedelta(3, "observations")
    >>> three_observations_log.get_name()
    '3 Observations'
    """

    _Observations = "o"

    # units for absolute times
    _absolute_units = ['ms', 's', 'h', 'm', 'd', 'w']
    _relative_units = ['mo', 'Y']

    _readable_units = {
        "ms": "Milliseconds",
        "s": "Seconds",
        "h": "Hours",
        "m": "Minutes",
        "d": "Days",
        "o": "Observations",
        "w": "Weeks",
        "Y": "Years",
        "mo": "Months"
    }

    _readable_to_unit = {v.lower(): k for k, v in _readable_units.items()}

    def __init__(self, value, unit=None, delta_obj=None):
        """
        Args:
            value (float, str) : Value of timedelta, or string providing
                both unit and value.
            unit (str) : Unit of time delta.
        """

        # TODO: check if value is int or float
        if is_string(value):
            from featuretools.utils.wrangle import _check_timedelta
            td = _check_timedelta(value)
            self.times = td.times
        elif isinstance(value, dict):
            self.times = value
        else:
            # elif isinstance(value, (int, float)) and isinstance(unit, str):
            self.times = {unit: value}
        self.fixed_units = dict()
        for unit, value in self.times.items():
            unit = self._check_unit_plural(unit)
            # assert unit in self._readable_units or unit in self._readable_to_unit
            if unit in self._readable_to_unit:
                unit = self._readable_to_unit[unit]
            self.fixed_units[unit] = value
        self.times = self.fixed_units
        if delta_obj is not None:
            self.delta_obj = delta_obj
        else:
            self.delta_obj = self.get_unit_type()

    @classmethod
    def from_dictionary(cls, dictionary):
        all_units = dict()
        dict_units = dictionary['unit']
        dict_values = dictionary['value']
        if isinstance(dict_units, str) and isinstance(dict_values, (int, float)):
            return cls({dict_units: dict_values})
        else:
            for i in range(len(dict_units)):
                all_units[dict_units[i]] = dict_values[i]
            return cls(all_units)

    @classmethod
    def make_singular(cls, s):
        if len(s) > 1 and s.endswith('s'):
            return s[:-1]
        return s

    @classmethod
    def _check_unit_plural(cls, s):
        if len(s) > 2 and not s.endswith('s'):
            return (s + 's').lower()
        elif len(s) > 1:
            return s.lower()
        return s

    def get_unit_type(self):
        all_units = list(self.times.keys())
        if self._Observations in all_units:
            return None
        elif self.is_absolute() and len(all_units) == 1:
            return pd.Timedelta(self.times[all_units[0]], all_units[0])
        else:
            readable_times = self.lower_readable_times()
            return relativedelta(**readable_times)

    def lower_readable_times(self):
        readable_times = dict()
        for unit, value in self.times.items():
            readable_unit = self._readable_units[unit].lower()
            readable_times[readable_unit] = value
        return readable_times

    def get_name(self):
        all_units = list(self.times.keys())
        if len(all_units) == 1:
            return "{} {}".format(self.times[all_units[0]], self._readable_units[all_units[0]])
        final_str = ""
        for unit, value in self.times.items():
            if value == 1:
                unit = self.make_singular(unit)
            final_str += "{} {} ".format(value, self._readable_units[unit])
        return final_str[:-1]

    # @property
    # def readable_unit(self):
    #     if self._original_unit is not None:
    #         return self._readable_units[self._original_unit]
    #     elif self.unit in self._readable_units.keys():
    #         return self._readable_units[self.unit]
    #     else:
    #         return self.unit

    def get_arguments(self):
        units = list()
        values = list()
        for unit, value in self.times.items():
            units.append(unit)
            values.append(value)
        if len(units) == 1:
            return {'unit': units[0], 'value': values[0]}
        else:
            return {'unit': units, 'value': values}

    def is_absolute(self):
        for unit in list(self.times.keys()):
            if unit not in self._absolute_units:
                return False
        return True

    def has_no_observations(self):
        for unit in list(self.times.keys()):
            if unit in self._Observations:
                return False
        return True

    def has_multiple_units(self):
        if isinstance(self.unit, list) and isinstance(self.value, list):
            return True
        else:
            return False

    def __eq__(self, other):
        if not isinstance(other, Timedelta):
            return False

        return (self.times == other.times)

    def __neg__(self):
        """Negate the timedelta"""
        new_times = dict()
        for unit, value in self.times.items():
            new_times[unit] = -value
        if self.delta_obj is not None:
            return Timedelta(new_times, delta_obj=-self.delta_obj)
        else:
            return Timedelta(new_times)

    def __radd__(self, time):
        """Add the Timedelta to a timestamp value"""
        if self._Observations not in self.times.keys():
            return time + self.delta_obj
        else:
            raise Exception("Invalid unit")

    def __rsub__(self, time):
        """Subtract the Timedelta from a timestamp value"""
        if self._Observations not in self.times.keys():
            return time - self.delta_obj
        else:
            raise Exception("Invalid unit")
