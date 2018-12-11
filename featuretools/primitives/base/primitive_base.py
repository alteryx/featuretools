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
    Categorical,
    Datetime,
    DatetimeTimeIndex,
    Id,
    Numeric,
    NumericTimeIndex,
    Variable
)

logger = logging.getLogger('featuretools')


class PrimitiveBase(object):
    """Base class for all features."""
    #: (str): Name of backend function used to compute this feature
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
    #: (:class:`.PrimitiveBase`): Feature to condition this feature by in
    # computation (e.g. take the Count of products where the product_id is
    # "basketball".)
    where = None
    #: (bool): If True, allow where clauses in DFS
    allow_where = False
    #: (str or :class:`.Timedelta`): Use only some amount of previous data from
    # each time point during calculation
    use_previous = None
    #: (int): Maximum number of features in the largest chain proceeding
    # downward from this feature's base features.
    max_stack_depth = None
    rolling_function = False
    #: (bool): If True, feature will expand into multiple values during
    # calculation
    expanding = False
    _name = None
    # whitelist of primitives can have this primitive in input_types
    base_of = None
    # blacklist of primitives can have this primitive in input_types
    base_of_exclude = None
    # (bool) If True will only make one feature per unique set of base features
    commutative = False
    # (bool) If True, feature function depends on all values of entity
    #   (and will receive these values as input, regardless of specified instance ids)
    uses_full_entity = False

    def __init__(self, entity, base_features, **kwargs):
        assert all(isinstance(f, PrimitiveBase) for f in base_features), \
            "All base features must be features"
        if len(set([bf.hash() for bf in base_features])) != len(base_features):
            raise ValueError(u"Duplicate base features ({}): {}".format(
                self.__class__, base_features))

        self.entity_id = entity.id
        self.entityset = entity.entityset.metadata

        # P TODO: where should this logic go?
        # not all primitives support use previous so doesn't make sense to have
        # in base
        if self.use_previous:
            self.use_previous = _check_timedelta(self.use_previous)
            assert len(self.base_features) > 0
            time_index = self.base_features[0].entity.time_index
            time_col = self.base_features[0].entity[time_index]
            assert time_index is not None, ("Use previous can only be defined "
                                            "on entities with a time index")
            assert _check_time_against_column(self.use_previous, time_col)

        self.base_features = base_features
        # variable type can be declared or inferred from first base feature
        self.additional_attributes = kwargs

        assert self._check_input_types(), ("Provided inputs don't match input "
                                           "type requirements")
        super(PrimitiveBase, self).__init__(**kwargs)

    @property
    def entity(self):
        """Entity this feature belongs too"""
        return self.entityset[self.entity_id]

    # P TODO: this should get refactored to return_type
    @property
    def variable_type(self):
        feature = self
        return_type = self.return_type

        while return_type is None:
            # get return type of first base feature
            base_feature = feature.base_features[0]
            return_type = base_feature.return_type

            # only the original time index should exist
            # so make this feature's return type just a Datetime
            if return_type == DatetimeTimeIndex:
                return_type = Datetime
            elif return_type == NumericTimeIndex:
                return_type = Numeric

            # direct features should keep the Id return type, but all other features should get
            # converted to Categorical
            if not isinstance(feature, DirectFeature) and return_type == Id:
                return_type = Categorical

            feature = base_feature

        return return_type

    @property
    def base_hashes(self):
        """Hashes of the base features"""
        return [f.hash() for f in self.base_features]

    def _check_feature(self, feature):
        if isinstance(feature, Variable):
            return IdentityFeature(feature)
        elif isinstance(feature, PrimitiveBase):
            return feature
        raise Exception("Not a feature")

    def __repr__(self):
        ret = "<Feature: %s>" % (self.get_name())

        # encode for python 2
        if type(ret) != str:
            ret = ret.encode("utf-8")

        return ret

    def hash(self):
        return hash(self.get_name() + self.entity.id)

    def __hash__(self):
        # logger.warning("To hash a feature, use feature.hash()")
        return self.hash()

    def __eq__(self, other_feature_or_val):
        """Compares to other_feature_or_val by equality

        See also:
            :meth:`PrimitiveBase.equal_to`
        """
        from featuretools.primitives import Equals
        return Equals(self, other_feature_or_val)

    def __ne__(self, other_feature_or_val):
        """Compares to other_feature_or_val by non-equality

        See also:
            :meth:`PrimitiveBase.not_equal_to`
        """
        from featuretools.primitives import NotEquals
        return NotEquals(self, other_feature_or_val)

    def __gt__(self, other_feature_or_val):
        """Compares if greater than other_feature_or_val

        See also:
            :meth:`PrimitiveBase.GT`
        """
        from featuretools.primitives import GreaterThan
        return GreaterThan(self, other_feature_or_val)

    def __ge__(self, other_feature_or_val):
        """Compares if greater than or equal to other_feature_or_val

        See also:
            :meth:`PrimitiveBase.greater_than_equal_to`
        """
        from featuretools.primitives import GreaterThanEqualTo
        return GreaterThanEqualTo(self, other_feature_or_val)

    def __lt__(self, other_feature_or_val):
        """Compares if less than other_feature_or_val

        See also:
            :meth:`PrimitiveBase.less_than`
        """
        from featuretools.primitives import LessThan
        return LessThan(self, other_feature_or_val)

    def __le__(self, other_feature_or_val):
        """Compares if less than or equal to other_feature_or_val

        See also:
            :meth:`PrimitiveBase.less_than_equal_to`
        """
        from featuretools.primitives import LessThanEqualTo
        return LessThanEqualTo(self, other_feature_or_val)

    def __add__(self, other_feature_or_val):
        """Add other_feature_or_val"""
        from featuretools.primitives import Add
        return Add(self, other_feature_or_val)

    def __radd__(self, other):
        from featuretools.primitives import Add
        return Add(other, self)

    def __sub__(self, other_feature_or_val):
        """Subtract other_feature_or_val

        See also:
            :meth:`PrimitiveBase.subtract`
        """
        from featuretools.primitives import Subtract
        return Subtract(self, other_feature_or_val)

    def __rsub__(self, other):
        from featuretools.primitives import Subtract
        return Subtract(other, self)

    def __div__(self, other_feature_or_val):
        """Divide by other_feature_or_val

        See also:
            :meth:`PrimitiveBase.divide`
        """
        from featuretools.primitives import Divide
        return Divide(self, other_feature_or_val)

    def __truediv__(self, other_feature_or_val):
        return self.__div__(other_feature_or_val)

    def __rtruediv__(self, other_feature_or_val):
        from featuretools.primitives import Divide
        return Divide(other_feature_or_val, self)

    def __rdiv__(self, other_feature_or_val):
        from featuretools.primitives import Divide
        return Divide(other_feature_or_val, self)

    def __mul__(self, other_feature_or_val):
        """Multiply by other_feature_or_val

        See also:
            :meth:`PrimitiveBase.multiply`
        """
        from featuretools.primitives import Multiply
        return Multiply(self, other_feature_or_val)

    def __rmul__(self, other):
        from featuretools.primitives import Multiply
        return Multiply(other, self)

    def __mod__(self, other_feature_or_val):
        """Take modulus of other_feature_or_val

        See also:
            :meth:`PrimitiveBase.modulo`
        """
        from featuretools.primitives import Mod
        return Mod(self, other_feature_or_val)

    def __and__(self, other):
        return self.AND(other)

    def __rand__(self, other):
        from featuretools.primitives import And
        return And(other, self)

    def __or__(self, other):
        return self.OR(other)

    def __ror__(self, other):
        from featuretools.primitives import Or
        return Or(other, self)

    def __not__(self, other):
        return self.NOT(other)

    def __abs__(self):
        from featuretools.primitives import Absolute
        return Absolute(self)

    def __neg__(self):
        from featuretools.primitives import Negate
        return Negate(self)

    def AND(self, other_feature):
        """Logical AND with other_feature"""
        from featuretools.primitives import And
        return And(self, other_feature)

    def OR(self, other_feature):
        """Logical OR with other_feature"""
        from featuretools.primitives import Or
        return Or(self, other_feature)

    def NOT(self):
        """Creates inverse of feature"""
        from featuretools.primitives import Not
        from featuretools.primitives import Compare
        if isinstance(self, Compare):
            return self.invert()
        return Not(self)

    def LIKE(self, like_string, case_sensitive=False):
        from featuretools.primitives import Like
        return Like(self, like_string,
                    case_sensitive=case_sensitive)

    def isin(self, list_of_output):
        from featuretools.primitives import IsIn
        return IsIn(self, list_of_outputs=list_of_output)

    def is_null(self):
        """Compares feature to null by equality"""
        from featuretools.primitives import IsNull
        return IsNull(self)

    def __invert__(self):
        return self.NOT()

    def rename(self, name):
        """Rename Feature, returns copy"""
        feature_copy = self.copy()
        feature_copy._name = name
        return feature_copy

    def copy(self):
        """Return copy of feature"""
        original_attrs = {}
        copied_attrs = {}
        for k, v in self.__dict__.items():
            list_like = False
            to_check = v
            if isinstance(v, (list, set, tuple)) and len(v):
                to_check = list(v)[0]
                list_like = True
            if isinstance(to_check, PrimitiveBase):
                if list_like:
                    copied_attrs[k] = [f.copy() for f in v]
                    original_attrs[k] = [f.copy() for f in v]
                else:
                    copied_attrs[k] = v.copy()
                    original_attrs[k] = v.copy()
                setattr(self, k, None)
            elif isinstance(to_check, (Variable, Entity, EntitySet)):
                copied_attrs[k] = v
                original_attrs[k] = v
                setattr(self, k, None)
        copied = copy.deepcopy(self)
        for k, v in copied_attrs.items():
            setattr(copied, k, v)
        for k, v in original_attrs.items():
            setattr(self, k, v)
        return copied

    def get_name(self):
        if self._name is not None:
            return self._name
        return self.generate_name()

    def get_function(self):
        raise NotImplementedError("Implement in subclass")

    def get_dependencies(self, deep=False, ignored=None, copy=True):
        """Returns features that are used to calculate this feature

        ..note::

            If you only want the features that make up the input to the feature
            function use the base_features attribute instead.


        """
        deps = []

        for d in self.base_features[:]:
            deps += [d]

        if self.where:
            deps += [self.where]

        # if self.use_previous and self.use_previous.is_absolute():
            # entity = self.entity
            # time_var = IdentityFeature(entity[entity.time_index])
            # deps += [time_var]

        if ignored is None:
            ignored = set([])
        deps = [d for d in deps if d.hash() not in ignored]

        if deep:
            for dep in deps[:]:  # copy so we don't modify list we iterate over
                deep_deps = dep.get_dependencies(deep, ignored)
                deps += deep_deps

        return deps

    def get_deep_dependencies(self, ignored=None):
        return self.get_dependencies(deep=True, ignored=ignored)

    def get_depth(self, stop_at=None):
        """Returns depth of feature"""
        max_depth = 0
        stop_at_hash = set()
        if stop_at is not None:
            stop_at_hash = set([i.hash() for i in stop_at])
        if (stop_at is not None and
                self.hash() in stop_at_hash):
            return 0
        for dep in self.get_deep_dependencies(ignored=stop_at_hash):
            max_depth = max(dep.get_depth(stop_at=stop_at),
                            max_depth)
        return max_depth + 1

    def _check_input_types(self):
        if len(self.base_features) == 0:
            return True

        input_types = self.input_types
        if input_types is not None:
            if type(self.input_types[0]) != list:
                input_types = [input_types]

            for t in input_types:
                zipped = list(zip(t, self.base_features))
                if all([issubclass(f.variable_type, v) for v, f in zipped]):
                    return True
        else:
            return True

        return False


class IdentityFeature(PrimitiveBase):
    """Feature for entity that is equivalent to underlying variable"""

    def __init__(self, variable):
        # TODO: perhaps we can change the attributes of this class to
        # just entityset reference to original variable object
        entity_id = variable.entity_id
        self.variable = variable.entityset.metadata[entity_id][variable.id]
        self.return_type = type(variable)
        self.base_feature = None
        super(IdentityFeature, self).__init__(variable.entity, [])

    def generate_name(self):
        return self.variable.name

    def get_depth(self, stop_at=None):
        return 0


class DirectFeature(PrimitiveBase):
    """Feature for child entity that inherits
        a feature value from a parent entity"""
    input_types = [Variable]
    return_type = None

    def __init__(self, base_feature, child_entity):
        base_feature = self._check_feature(base_feature)
        if base_feature.expanding:
            self.expanding = True

        path = child_entity.entityset.find_forward_path(child_entity.id, base_feature.entity.id)
        if len(path) > 1:
            parent_entity_id = path[1].child_entity.id
            parent_entity = child_entity.entityset[parent_entity_id]
            parent_feature = DirectFeature(base_feature, parent_entity)
        else:
            parent_feature = base_feature

        self.parent_entity = parent_feature.entity
        self._variable_type = parent_feature.variable_type
        super(DirectFeature, self).__init__(child_entity, [parent_feature])

    @property
    def default_value(self):
        return self.base_features[0].default_value

    @property
    def variable(self):
        return getattr(self.base_features[0], 'variable', None)

    def generate_name(self):
        return u"%s.%s" % (self.parent_entity.id,
                           self.base_features[0].get_name())


class Feature(PrimitiveBase):
    """
    Alias for IdentityFeature and DirectFeature depending on arguments
    """

    def __new__(self, feature_or_var, entity=None):
        if entity is None:
            assert isinstance(feature_or_var, (Variable))
            return IdentityFeature(feature_or_var)

        assert isinstance(feature_or_var, (Variable, PrimitiveBase))
        assert isinstance(entity, Entity)

        if feature_or_var.entity.id == entity.id:
            return IdentityFeature(entity)

        return DirectFeature(feature_or_var, entity)
