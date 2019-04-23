from __future__ import division

from builtins import object

import numpy as np
import pandas as pd

from featuretools.utils import is_string


class Variable(object):
    """Represent a variable in an entity

    A Variable is analogous to a column in table in a relational database

    Args:
        id (str) : Id of variable. Must match underlying data in Entity
            it belongs to.
        entity (:class:`.Entity`) : Entity this variable belongs to.
        name (str, optional) : Variable name. Defaults to id.

    See Also:
        :class:`.Entity`, :class:`.Relationship`, :class:`.BaseEntitySet`
    """
    type_string = None
    _default_pandas_dtype = object

    def __init__(self, id, entity, name=None):
        assert is_string(id), "Variable id must be a string"
        self.id = id
        self._name = name
        self.entity_id = entity.id
        assert entity.entityset is not None, "Entity must contain reference to EntitySet"
        self.entity = entity
        self._interesting_values = None

    @property
    def entityset(self):
        return self.entity.entityset

    def __eq__(self, other):
        return isinstance(other, self.__class__) and \
            self.id == other.id and \
            self.entity_id == other.entity_id

    def __repr__(self):
        ret = u"<Variable: {} (dtype = {})>".format(self.name, self.type_string)

        # encode for python 2
        if type(ret) != str:
            ret = ret.encode("utf-8")

        return ret

    @classmethod
    def create_from(cls, variable):
        """Create new variable this type from existing

        Args:
            variable (Variable) : Existing variable to create from.

        Returns:
            :class:`.Variable` : new variable

        """
        v = cls(id=variable.id, name=variable.name, entity=variable.entity)
        return v

    @property
    def name(self):
        return self._name if self._name is not None else self.id

    @property
    def dtype(self):
        return self.type_string \
            if self.type_string is not None else "generic_type"

    @name.setter
    def name(self, name):
        self._name = name

    @property
    def interesting_values(self):
        return self._interesting_values

    @interesting_values.setter
    def interesting_values(self, interesting_values):
        self._interesting_values = interesting_values

    @property
    def series(self):
        return self.entity.df[self.id]

    def to_data_description(self):
        return {
            'id': self.id,
            'type': {
                'value': self.type_string,
            },
            'properties': {
                'name': self.name,
                'entity': self.entity.id,
                'interesting_values': self._interesting_values
            },
        }


class Unknown(Variable):
    pass


class Discrete(Variable):
    """Superclass representing variables that take on discrete values"""
    type_string = "discrete"

    def __init__(self, id, entity, name=None):
        super(Discrete, self).__init__(id, entity, name)
        self._interesting_values = []

    @property
    def interesting_values(self):
        return self._interesting_values

    @interesting_values.setter
    def interesting_values(self, values):
        seen = set()
        seen_add = seen.add
        self._interesting_values = [v for v in values
                                    if not (v in seen or seen_add(v))]


class Boolean(Variable):
    """Represents variables that take on one of two values

    Args:
        true_values (list) : List of valued true values. Defaults to [1, True, "true", "True", "yes", "t", "T"]
        false_values (list): List of valued false values. Defaults to [0, False, "false", "False", "no", "f", "F"]
    """
    type_string = "boolean"
    _default_pandas_dtype = bool

    def __init__(self,
                 id,
                 entity,
                 name=None,
                 true_values=None,
                 false_values=None):
        default = [1, True, "true", "True", "yes", "t", "T"]
        self.true_values = true_values or default
        default = [0, False, "false", "False", "no", "f", "F"]
        self.false_values = false_values or default
        super(Boolean, self).__init__(id, entity, name=name)

    def to_data_description(self):
        description = super(Boolean, self).to_data_description()
        description['type'].update({
            'true_values': self.true_values,
            'false_values': self.false_values
        })
        return description


class Categorical(Discrete):
    """Represents variables that can take an unordered discrete values

    Args:
        categories (list) : List of categories. If left blank, inferred from data.
    """
    type_string = "categorical"

    def __init__(self, id, entity, name=None, categories=None):
        self.categories = None or []
        super(Categorical, self).__init__(id, entity, name=name)

    def to_data_description(self):
        description = super(Categorical, self).to_data_description()
        description['type'].update({'categories': self.categories})
        return description


class Id(Categorical):
    """Represents variables that identify another entity"""
    type_string = "id"
    _default_pandas_dtype = int


class Ordinal(Discrete):
    """Represents variables that take on an ordered discrete value"""
    type_string = "ordinal"
    _default_pandas_dtype = int


class Numeric(Variable):
    """Represents variables that contain numeric values

    Args:
        range (list, optional) : List of start and end. Can use inf and -inf to represent infinity. Unconstrained if not specified.
        start_inclusive (bool, optional) : Whether or not range includes the start value.
        end_inclusive (bool, optional) : Whether or not range includes the end value

    Attributes:
        max (float)
        min (float)
        std (float)
        mean (float)
    """
    type_string = "numeric"
    _default_pandas_dtype = float

    def __init__(self,
                 id,
                 entity,
                 name=None,
                 range=None,
                 start_inclusive=True,
                 end_inclusive=False):
        self.range = None or []
        self.start_inclusive = start_inclusive
        self.end_inclusive = end_inclusive
        super(Numeric, self).__init__(id, entity, name=name)

    def to_data_description(self):
        description = super(Numeric, self).to_data_description()
        description['type'].update({
            'range': self.range,
            'start_inclusive': self.start_inclusive,
            'end_inclusive': self.end_inclusive,
        })
        return description


class Index(Variable):
    """Represents variables that uniquely identify an instance of an entity

    Attributes:
        count (int)
    """
    type_string = "index"
    _default_pandas_dtype = int


class Datetime(Variable):
    """Represents variables that are points in time

    Args:
        format (str): Python datetime format string documented `here <http://strftime.org/>`_.
    """
    type_string = "datetime"
    _default_pandas_dtype = np.datetime64

    def __init__(self, id, entity, name=None, format=None):
        self.format = format
        super(Datetime, self).__init__(id, entity, name=name)

    def __repr__(self):
        ret = u"<Variable: {} (dtype: {}, format: {})>".format(self.name, self.type_string, self.format)

        # encode for python 2
        if type(ret) != str:
            ret = ret.encode("utf-8")

        return ret

    def to_data_description(self):
        description = super(Datetime, self).to_data_description()
        description['type'].update({'format': self.format})
        return description


class TimeIndex(Variable):
    """Represents time index of entity"""
    type_string = "time_index"
    _default_pandas_dtype = np.datetime64


class NumericTimeIndex(TimeIndex, Numeric):
    """Represents time index of entity that is numeric"""
    type_string = "numeric_time_index"
    _default_pandas_dtype = float


class DatetimeTimeIndex(TimeIndex, Datetime):
    """Represents time index of entity that is a datetime"""
    type_string = "datetime_time_index"
    _default_pandas_dtype = np.datetime64


class Timedelta(Variable):
    """Represents variables that are timedeltas

    Args:
        range (list, optional) : List of start and end of allowed range in seconds. Can use inf and -inf to represent infinity. Unconstrained if not specified.
        start_inclusive (bool, optional) : Whether or not range includes the start value.
        end_inclusive (bool, optional) : Whether or not range includes the end value
    """
    type_string = "timedelta"
    _default_pandas_dtype = np.timedelta64

    def __init__(self,
                 id,
                 entity,
                 name=None,
                 range=None,
                 start_inclusive=True,
                 end_inclusive=False):
        self.range = range or []
        self.start_inclusive = start_inclusive
        self.end_inclusive = end_inclusive
        super(Timedelta, self).__init__(id, entity, name=name)

    def to_data_description(self):
        description = super(Timedelta, self).to_data_description()
        description['type'].update({
            'range': self.range,
            'start_inclusive': self.start_inclusive,
            'end_inclusive': self.end_inclusive,
        })
        return description


class Text(Variable):
    """Represents variables that are arbitary strings"""
    type_string = "text"
    _default_pandas_dtype = str


class PandasTypes(object):
    _all = 'all'
    _categorical = 'category'
    _pandas_datetimes = ['datetime64[ns]', 'datetime64[ns, tz]']
    _pandas_timedeltas = ['Timedelta']
    _pandas_numerics = ['int16', 'int32', 'int64',
                        'float16', 'float32', 'float64']


class LatLong(Variable):
    """Represents an ordered pair (Latitude, Longitude)
    To make a latlong in a dataframe do
    data['latlong'] = data[['latitude', 'longitude']].apply(tuple, axis=1)
    """
    type_string = "latlong"


class ZIPCode(Categorical):
    """Represents a postal address in the United States.
    Consists of a series of digits which are casts as
    string. Five digit and 9 digit zipcodes are supported.
    """
    type_string = "zipcode"
    _default_pandas_dtype = str


class IPAddress(Variable):
    """Represents a computer network address. Represented
    in dotted-decimal notation. IPv4 and IPv6 are supported.
    """
    type_string = "ip"
    _default_pandas_dtype = str


class EmailAddress(Variable):
    """Represents an email box to which email message are sent.
    Consits of a local-part, an @ symbol, and a domain.
    """
    type_string = "email"
    _default_pandas_dtype = str


class URL(Variable):
    """Represents a valid web url (with or without http/www)"""
    type_string = "url"
    _default_pandas_dtype = str


class PhoneNumber(Variable):
    """Represents any valid phone number.
    Can be with/without parenthesis.
    Can be with/without area/country codes.
    """
    type_string = "phone_number"
    _default_pandas_dtype = str


class DateOfBirth(Datetime):
    """Represents a date of birth as a datetime"""
    type_string = "date_of_birth"
    _default_pandas_dtype = np.datetime64


class CountryCode(Categorical):
    """Represents an ISO-3166 standard country code.
    ISO 3166-1 (countries) are supported. These codes
    should be in the Alpha-2 format.
    e.g. United States of America = US
    """
    type_string = "country_code"
    _default_pandas_dtype = str


class SubRegionCode(Categorical):
    """Represents an ISO-3166 standard sub-region code.
    ISO 3166-2 codes (sub-regions are supported. These codes
    should be in the Alpha-2 format.
    e.g. United States of America, Arizona = US-AZ
    """
    type_string = "subregion_code"
    _default_pandas_dtype = str


class FilePath(Variable):
    """Represents a valid filepath, absolute or relative"""
    type_string = "filepath"
    _default_pandas_dtype = str


ALL_VARIABLE_TYPES = [Datetime, Numeric, Timedelta,
                      Categorical, Text, Ordinal,
                      Boolean, LatLong, ZIPCode, IPAddress,
                      EmailAddress, URL, PhoneNumber,
                      DateOfBirth, CountryCode, SubRegionCode,
                      FilePath]


DEFAULT_DTYPE_VALUES = {
    np.datetime64: pd.Timestamp.now(),
    int: 0,
    float: 0.1,
    np.timedelta64: pd.Timedelta('1d'),
    object: 'object',
    bool: True,
    str: 'test'
}
