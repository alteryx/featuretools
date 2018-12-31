from __future__ import absolute_import

import copy
import logging
from builtins import zip

from numpy import nan

from featuretools.entityset import Entity, EntitySet
from featuretools.utils.wrangle import (
    _check_time_against_column,
    _check_timedelta
)
from featuretools.variable_types import (
    Datetime,
    Id,
    Numeric,
    NumericTimeIndex,
    Variable
)

logger = logging.getLogger('featuretools')

class PrimitiveBase(object):
    """Base class for all primitives."""
    #: (str): Name of the primitive
    name = None
    #: (list): Variable types of inputs
    input_types = None
    #: (:class:`.Variable`): variable type of return
    return_type = None
    #: Default value this feature returns if no data found. deafults to np.nan
    default_value = nan
    #: (bool): True if feature needs to know what the current calculation time
    # is (provided to computational backend as "time_last")
    uses_calc_time = False
    #: (bool): If True, allow where clauses in DFS
    allow_where = False
    #: (int): Maximum number of features in the largest chain proceeding
    # downward from this feature's base features.
    max_stack_depth = None
    #: (bool): If True, feature will expand into multiple values during
    # calculation
    expanding = False
    # whitelist of primitives can have this primitive in input_types
    base_of = None
    # blacklist of primitives can have this primitive in input_types
    base_of_exclude = None
    # (bool) If True will only make one feature per unique set of base features
    commutative = False




#     def __eq__(self, other_feature_or_val):
#         """Compares to other_feature_or_val by equality

#         See also:
#             :meth:`PrimitiveBase.equal_to`
#         """
#         from featuretools.primitives import Equals
#         return Equals(self, other_feature_or_val)

#     def __ne__(self, other_feature_or_val):
#         """Compares to other_feature_or_val by non-equality

#         See also:
#             :meth:`PrimitiveBase.not_equal_to`
#         """
#         from featuretools.primitives import NotEquals
#         return NotEquals(self, other_feature_or_val)

#     def __gt__(self, other_feature_or_val):
#         """Compares if greater than other_feature_or_val

#         See also:
#             :meth:`PrimitiveBase.GT`
#         """
#         from featuretools.primitives import GreaterThan
#         return GreaterThan(self, other_feature_or_val)

#     def __ge__(self, other_feature_or_val):
#         """Compares if greater than or equal to other_feature_or_val

#         See also:
#             :meth:`PrimitiveBase.greater_than_equal_to`
#         """
#         from featuretools.primitives import GreaterThanEqualTo
#         return GreaterThanEqualTo(self, other_feature_or_val)

#     def __lt__(self, other_feature_or_val):
#         """Compares if less than other_feature_or_val

#         See also:
#             :meth:`PrimitiveBase.less_than`
#         """
#         from featuretools.primitives import LessThan
#         return LessThan(self, other_feature_or_val)

#     def __le__(self, other_feature_or_val):
#         """Compares if less than or equal to other_feature_or_val

#         See also:
#             :meth:`PrimitiveBase.less_than_equal_to`
#         """
#         from featuretools.primitives import LessThanEqualTo
#         return LessThanEqualTo(self, other_feature_or_val)

#     def __add__(self, other_feature_or_val):
#         """Add other_feature_or_val"""
#         from featuretools.primitives import Add
#         return Add(self, other_feature_or_val)

#     def __radd__(self, other):
#         from featuretools.primitives import Add
#         return Add(other, self)

#     def __sub__(self, other_feature_or_val):
#         """Subtract other_feature_or_val

#         See also:
#             :meth:`PrimitiveBase.subtract`
#         """
#         from featuretools.primitives import Subtract
#         return Subtract(self, other_feature_or_val)

#     def __rsub__(self, other):
#         from featuretools.primitives import Subtract
#         return Subtract(other, self)

#     def __div__(self, other_feature_or_val):
#         """Divide by other_feature_or_val

#         See also:
#             :meth:`PrimitiveBase.divide`
#         """
#         from featuretools.primitives import Divide
#         return Divide(self, other_feature_or_val)

#     def __truediv__(self, other_feature_or_val):
#         return self.__div__(other_feature_or_val)

#     def __rtruediv__(self, other_feature_or_val):
#         from featuretools.primitives import Divide
#         return Divide(other_feature_or_val, self)

#     def __rdiv__(self, other_feature_or_val):
#         from featuretools.primitives import Divide
#         return Divide(other_feature_or_val, self)

#     def __mul__(self, other_feature_or_val):
#         """Multiply by other_feature_or_val

#         See also:
#             :meth:`PrimitiveBase.multiply`
#         """
#         from featuretools.primitives import Multiply
#         return Multiply(self, other_feature_or_val)

#     def __rmul__(self, other):
#         from featuretools.primitives import Multiply
#         return Multiply(other, self)

#     def __mod__(self, other_feature_or_val):
#         """Take modulus of other_feature_or_val

#         See also:
#             :meth:`PrimitiveBase.modulo`
#         """
#         from featuretools.primitives import Mod
#         return Mod(self, other_feature_or_val)

#     def __and__(self, other):
#         return self.AND(other)

#     def __rand__(self, other):
#         from featuretools.primitives import And
#         return And(other, self)

#     def __or__(self, other):
#         return self.OR(other)

#     def __ror__(self, other):
#         from featuretools.primitives import Or
#         return Or(other, self)

#     def __not__(self, other):
#         return self.NOT(other)

#     def __abs__(self):
#         from featuretools.primitives import Absolute
#         return Absolute(self)

#     def __neg__(self):
#         from featuretools.primitives import Negate
#         return Negate(self)

#     def AND(self, other_feature):
#         """Logical AND with other_feature"""
#         from featuretools.primitives import And
#         return And(self, other_feature)

#     def OR(self, other_feature):
#         """Logical OR with other_feature"""
#         from featuretools.primitives import Or
#         return Or(self, other_feature)

#     def NOT(self):
#         """Creates inverse of feature"""
#         from featuretools.primitives import Not
#         from featuretools.primitives import Compare
#         if isinstance(self, Compare):
#             return self.invert()
#         return Not(self)

#     def LIKE(self, like_string, case_sensitive=False):
#         from featuretools.primitives import Like
#         return Like(self, like_string,
#                     case_sensitive=case_sensitive)

#     def isin(self, list_of_output):
#         from featuretools.primitives import IsIn
#         return IsIn(self, list_of_outputs=list_of_output)

#     def is_null(self):
#         """Compares feature to null by equality"""
#         from featuretools.primitives import IsNull
#         return IsNull(self)

#     def __invert__(self):
#         return self.NOT()



# class Feature(PrimitiveBase):
#     """
#     Alias for IdentityFeature and DirectFeature depending on arguments
#     """

#     def __new__(self, feature_or_var, entity=None):
#         if entity is None:
#             assert isinstance(feature_or_var, (Variable))
#             return IdentityFeature(feature_or_var)

#         assert isinstance(feature_or_var, (Variable, PrimitiveBase))
#         assert isinstance(entity, Entity)

#         if feature_or_var.entity.id == entity.id:
#             return IdentityFeature(entity)

#         return DirectFeature(feature_or_var, entity)
