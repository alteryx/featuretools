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
    _dtype_repr = None
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
        ret = u"<Variable: {} (dtype = {})>".format(self.name, self.dtype)

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
        return self._dtype_repr \
            if self._dtype_repr is not None else "generic_type"

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

    def create_metadata_dict(self):
        return {
            'dtype_repr': self._dtype_repr,
            'entity': self.entity.id,
            'id': self.id,
            'name': self.name,
            'interesting_values': self._interesting_values
        }


class Unknown(Variable):
    pass


class Discrete(Variable):
    """Superclass representing variables that take on discrete values"""
    _dtype_repr = "discrete"

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
    """Represents variables that take on one of two values"""
    _dtype_repr = "boolean"
    _default_pandas_dtype = bool


class Categorical(Discrete):
    """Represents variables that can take an unordered discrete values"""
    _dtype_repr = "categorical"


class Id(Categorical):
    """Represents variables that identify another entity"""
    _dtype_repr = "id"
    _default_pandas_dtype = int


class Ordinal(Discrete):
    """Represents variables that take on an ordered discrete value"""
    _dtype_repr = "ordinal"
    _default_pandas_dtype = int


class Numeric(Variable):
    """Represents variables that contain numeric values

    Attributes:
        max (float)
        min (float)
        std (float)
        mean (float)
    """
    _dtype_repr = "numeric"
    _default_pandas_dtype = float


class Index(Variable):
    """Represents variables that uniquely identify an instance of an entity

    Attributes:
        count (int)
    """
    _dtype_repr = "index"
    _default_pandas_dtype = int


class Datetime(Variable):
    """Represents variables that are points in time"""
    _dtype_repr = "datetime"
    _default_pandas_dtype = np.datetime64

    def __init__(self, id, entity, format=None, name=None):
        self.format = format
        super(Datetime, self).__init__(id, entity, name)

    def __repr__(self):
        ret = u"<Variable: {} (dtype: {}, format: {})>".format(self.name, self.dtype, self.format)

        # encode for python 2
        if type(ret) != str:
            ret = ret.encode("utf-8")

        return ret


class TimeIndex(Variable):
    """Represents time index of entity"""
    _dtype_repr = "time_index"
    _default_pandas_dtype = np.datetime64


class NumericTimeIndex(TimeIndex, Numeric):
    """Represents time index of entity that is numeric"""
    _dtype_repr = "numeric_time_index"
    _default_pandas_dtype = float


class DatetimeTimeIndex(TimeIndex, Datetime):
    """Represents time index of entity that is a datetime"""
    _dtype_repr = "datetime_time_index"
    _default_pandas_dtype = np.datetime64


class Timedelta(Variable):
    """Represents variables that are timedeltas"""
    _dtype_repr = "timedelta"
    _default_pandas_dtype = np.timedelta64


class Text(Variable):
    """Represents variables that are arbitary strings"""
    _dtype_repr = "text"
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
    _dtype_repr = "latlong"


ALL_VARIABLE_TYPES = [Datetime, Numeric, Timedelta,
                      Categorical, Text, Ordinal,
                      Boolean, LatLong]


DEFAULT_DTYPE_VALUES = {
    np.datetime64: pd.Timestamp.now(),
    int: 0,
    float: 0.1,
    np.timedelta64: pd.Timedelta('1d'),
    object: 'object',
    bool: True,
    str: 'test'
}
