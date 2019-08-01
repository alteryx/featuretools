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

    # units for absolute times
    _absolute_units = ['ms', 's', 'h', 'm', 'd']
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
        'mo': "Months"
    }

    _convert_to_days = {
        "w": 7
    }

    _readable_to_unit = {v.lower(): k for k, v in _readable_units.items()}

    def __init__(self, value, unit=None, entity=None):
        """
        Args:
            value (float, str) : Value of timedelta, or string providing
                both unit and value.
            unit (str) : Unit of time delta.
            entity (str, optional) : Entity id to use if unit equals
                "observations".
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
        self.delta_obj = self.get_unit_type()

    @classmethod
    def from_dictionary(cls, dictionary):
        return cls(dictionary['value'],
                   unit=dictionary['unit'],
                   entity=dictionary['entity_id'])

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
        if self.unit == "o":
            return None
        elif self.unit in self._absolute_units:
            return pd.Timedelta(self.value, self.unit)
        else:
            unit = self.readable_unit.lower()
            return relativedelta(**{unit: self.value})

    def get_name(self):
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
                self.entity == other.entity)

    @property
    def readable_unit(self):
        if self._original_unit is not None:
            return self._readable_units[self._original_unit]
        return self._readable_units[self.unit]

    def get_delta_obj(self):
        return self.delta_obj

    def get_date_offset(self):
        return pd.DateOffset(**{self.unit: self.value})

    def view(self, unit):
        if self.is_absolute():
            return self.delta_obj.view(unit)
        else:
            raise Exception("Invalid unit")

    @property
    def value_in_seconds(self):
        if self.is_absolute():
            return self.delta_obj.total_seconds()
        else:
            raise Exception("Invalid unit")

    def get_arguments(self):
        return {
            'value': self._original_value(),
            'unit': self._original_unit or self.unit,
            'entity_id': self.entity
        }

    def _original_value(self):
        if self._original_unit:
            return self.value / self._convert_to_days[self._original_unit]
        else:
            return self.value

    def is_absolute(self):
        return self.unit in self._absolute_units

    def __neg__(self):
        """Negate the timedelta"""
        return Timedelta(-self.value, self.unit, self.entity)

    def __radd__(self, time):
        """Add the Timedelta to a timestamp value"""
        if self.unit != self._Observations:
            return time + self.delta_obj
        else:
            raise Exception("Invalid unit")

    def __rsub__(self, time):
        """Subtract the Timedelta from a timestamp value"""
        if self.unit != self._Observations:
            return time - self.delta_obj
        else:
            raise Exception("Invalid unit")
