from builtins import zip

from featuretools import Relationship, Timedelta, primitives
from featuretools.primitives.base import (
    AggregationPrimitive,
    PrimitiveBase,
    TransformPrimitive
)
from featuretools.primitives.utils import serialize_primitive
from featuretools.utils.wrangle import (
    _check_time_against_column,
    _check_timedelta
)
from featuretools.variable_types import (
    Categorical,
    Datetime,
    DatetimeTimeIndex,
    Discrete,
    Id,
    Numeric,
    NumericTimeIndex,
    Variable
)


class FeatureBase(object):
    def __init__(self, entity, base_features, relationship_path, primitive):
        """Base class for all features

        Args:
            entity (Entity): entity this feature is being calculated for
            base_features (list[FeatureBase]): list of base features for primitive
            primitive (:class:`.PrimitiveBase`): primitive to calculate. if not initialized when passed, gets initialized with no arguments
        """
        assert all(isinstance(f, FeatureBase) for f in base_features), \
            "All base features must be features"

        self.entity_id = entity.id
        self.entityset = entity.entityset.metadata

        self.base_features = base_features

        # initialize if not already initialized
        if not isinstance(primitive, PrimitiveBase):
            primitive = primitive()
        self.primitive = primitive

        self.relationship_path = relationship_path

        self._name = None

        assert self._check_input_types(), ("Provided inputs don't match input "
                                           "type requirements")

    @classmethod
    def from_dictionary(cls, arguments, entityset, dependencies, primitives_deserializer):
        raise NotImplementedError("Must define from_dictionary on FeatureBase subclass")

    def rename(self, name):
        """Rename Feature, returns copy"""
        feature_copy = self.copy()
        feature_copy._name = name
        return feature_copy

    def copy(self):
        raise NotImplementedError("Must define copy on FeatureBase subclass")

    def get_name(self):
        if not self._name:
            self._name = self.generate_name()
        return self._name

    def get_feature_names(self):
        n = self.number_output_features
        if n == 1:
            names = [self.get_name()]
        else:
            names = [self.get_name() + "__{}".format(i) for i in range(n)]
        return names

    def get_function(self):
        return self.primitive.get_function()

    def get_dependencies(self, deep=False, ignored=None, copy=True):
        """Returns features that are used to calculate this feature

        ..note::

            If you only want the features that make up the input to the feature
            function use the base_features attribute instead.


        """
        deps = []

        for d in self.base_features[:]:
            deps += [d]

        if hasattr(self, "where") and self.where:
            deps += [self.where]

        if ignored is None:
            ignored = set([])
        deps = [d for d in deps if d.hash() not in ignored]

        if deep:
            for dep in deps[:]:  # copy so we don't modify list we iterate over
                deep_deps = dep.get_dependencies(deep, ignored)
                deps += deep_deps

        return deps

    def get_depth(self, stop_at=None):
        """Returns depth of feature"""
        max_depth = 0
        stop_at_hash = set()
        if stop_at is not None:
            stop_at_hash = set([i.hash() for i in stop_at])
        if (stop_at is not None and
                self.hash() in stop_at_hash):
            return 0
        for dep in self.get_dependencies(deep=True, ignored=stop_at_hash):
            max_depth = max(dep.get_depth(stop_at=stop_at),
                            max_depth)
        return max_depth + 1

    def _check_input_types(self):
        if len(self.base_features) == 0:
            return True

        input_types = self.primitive.input_types
        if input_types is not None:
            if type(input_types[0]) != list:
                input_types = [input_types]

            for t in input_types:
                zipped = list(zip(t, self.base_features))
                if all([issubclass(f.variable_type, v) for v, f in zipped]):
                    return True
        else:
            return True

        return False

    @property
    def entity(self):
        """Entity this feature belongs too"""
        return self.entityset[self.entity_id]

    @property
    def number_output_features(self):
        return self.primitive.number_output_features

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

    @property
    def variable_type(self):
        feature = self
        variable_type = self.primitive.return_type

        while variable_type is None:
            # get variable_type of first base feature
            base_feature = feature.base_features[0]
            variable_type = base_feature.variable_type

            # only the original time index should exist
            # so make this feature's return type just a Datetime
            if variable_type == DatetimeTimeIndex:
                variable_type = Datetime
            elif variable_type == NumericTimeIndex:
                variable_type = Numeric

            # direct features should keep the Id return type, but all other features should get
            # converted to Categorical
            if not isinstance(feature, DirectFeature) and variable_type == Id:
                variable_type = Categorical

            feature = base_feature

        return variable_type

    @property
    def default_value(self):
        return self.primitive.default_value

    def get_arguments(self):
        raise NotImplementedError("Must define get_arguments on FeatureBase subclass")

    def to_dictionary(self):
        return {
            'type': type(self).__name__,
            'dependencies': [dep.unique_name() for dep in self.get_dependencies()],
            'arguments': self.get_arguments(),
        }

    def _handle_binary_comparision(self, other, Primitive, PrimitiveScalar):
        if isinstance(other, FeatureBase):
            return Feature([self, other], primitive=Primitive)

        return Feature([self], primitive=PrimitiveScalar(other))

    def __eq__(self, other):
        """Compares to other by equality"""
        return self._handle_binary_comparision(other, primitives.Equal, primitives.EqualScalar)

    def __ne__(self, other):
        """Compares to other by non-equality"""
        return self._handle_binary_comparision(other, primitives.NotEqual, primitives.NotEqualScalar)

    def __gt__(self, other):
        """Compares if greater than other"""
        return self._handle_binary_comparision(other, primitives.GreaterThan, primitives.GreaterThanScalar)

    def __ge__(self, other):
        """Compares if greater than or equal to other"""
        return self._handle_binary_comparision(other, primitives.GreaterThanEqualTo, primitives.GreaterThanEqualToScalar)

    def __lt__(self, other):
        """Compares if less than other"""
        return self._handle_binary_comparision(other, primitives.LessThan, primitives.LessThanScalar)

    def __le__(self, other):
        """Compares if less than or equal to other"""
        return self._handle_binary_comparision(other, primitives.LessThanEqualTo, primitives.LessThanEqualToScalar)

    def __add__(self, other):
        """Add other"""
        return self._handle_binary_comparision(other, primitives.AddNumeric, primitives.AddNumericScalar)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        """Subtract other"""
        return self._handle_binary_comparision(other, primitives.SubtractNumeric, primitives.SubtractNumericScalar)

    def __rsub__(self, other):
        return Feature([self], primitive=primitives.ScalarSubtractNumericFeature(other))

    def __div__(self, other):
        """Divide by other"""
        return self._handle_binary_comparision(other, primitives.DivideNumeric, primitives.DivideNumericScalar)

    def __truediv__(self, other):
        return self.__div__(other)

    def __rtruediv__(self, other):
        return self.__rdiv__(other)

    def __rdiv__(self, other):
        return Feature([self], primitive=primitives.DivideByFeature(other))

    def __mul__(self, other):
        """Multiply by other"""
        return self._handle_binary_comparision(other, primitives.MultiplyNumeric, primitives.MultiplyNumericScalar)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __mod__(self, other):
        """Take modulus of other"""
        return self._handle_binary_comparision(other, primitives.ModuloNumeric, primitives.ModuloNumericScalar)

    def __rmod__(self, other):
        return Feature([self], primitive=primitives.ModuloByFeature(other))

    def __and__(self, other):
        return self.AND(other)

    def __rand__(self, other):
        return Feature([other, self], primitive=primitives.And)

    def __or__(self, other):
        return self.OR(other)

    def __ror__(self, other):
        return Feature([other, self], primitive=primitives.Or)

    def __not__(self, other):
        return self.NOT(other)

    def __abs__(self):
        return Feature([self], primitive=primitives.Absolute)

    def __neg__(self):
        return Feature([self], primitive=primitives.Negate)

    def AND(self, other_feature):
        """Logical AND with other_feature"""
        return Feature([self, other_feature], primitive=primitives.And)

    def OR(self, other_feature):
        """Logical OR with other_feature"""
        return Feature([self, other_feature], primitive=primitives.Or)

    def NOT(self):
        """Creates inverse of feature"""
        return Feature([self], primitive=primitives.Not)

    def isin(self, list_of_output):
        return Feature([self], primitive=primitives.IsIn(list_of_outputs=list_of_output))

    def is_null(self):
        """Compares feature to null by equality"""
        return Feature([self], primitive=primitives.IsNull)

    def __invert__(self):
        return self.NOT()

    def unique_name(self):
        return u"%s.%s" % (self.entity_id, self.get_name())


class IdentityFeature(FeatureBase):
    """Feature for entity that is equivalent to underlying variable"""

    def __init__(self, variable):
        entity_id = variable.entity_id
        self.variable = variable.entityset.metadata[entity_id][variable.id]
        self.return_type = type(variable)
        super(IdentityFeature, self).__init__(entity=variable.entity,
                                              base_features=[],
                                              relationship_path=[],
                                              primitive=PrimitiveBase)

    @classmethod
    def from_dictionary(cls, arguments, entityset, dependencies, primitives_deserializer):
        entity_id = arguments['entity_id']
        variable_id = arguments['variable_id']
        variable = entityset[entity_id][variable_id]
        return cls(variable)

    def copy(self):
        """Return copy of feature"""
        return IdentityFeature(self.variable)

    def generate_name(self):
        return self.variable.name

    def get_depth(self, stop_at=None):
        return 0

    def get_arguments(self):
        return {
            'variable_id': self.variable.id,
            'entity_id': self.variable.entity_id,
        }

    @property
    def variable_type(self):
        return type(self.variable)


class DirectFeature(FeatureBase):
    """Feature for child entity that inherits
        a feature value from a parent entity"""
    input_types = [Variable]
    return_type = None

    def __init__(self, base_feature, child_entity, relationship_path=None):
        """relationship_path is a forward path from child to parent."""
        base_feature = _check_feature(base_feature)

        self.parent_entity = base_feature.entity

        relationship_path, self._path_is_unique = \
            self._handle_relationship_path(child_entity, relationship_path)

        super(DirectFeature, self).__init__(entity=child_entity,
                                            base_features=[base_feature],
                                            relationship_path=relationship_path,
                                            primitive=PrimitiveBase)

    def _handle_relationship_path(self, child_entity, relationship_path):
        if relationship_path:
            first_child = relationship_path[0].child_entity
            assert child_entity == first_child, \
                'child_entity must match the first relationship'

            assert self.parent_entity == relationship_path[-1].parent_entity, \
                'Base feature must be defined on the entity at the end of relationship_path'

            path_is_unique = child_entity.entityset \
                .has_unique_forward_path(child_entity.id, self.parent_entity.id)
        else:
            relationship_path = _find_path(child_entity.id,
                                           self.parent_entity.id,
                                           child_entity.entityset)
            path_is_unique = True

        return relationship_path, path_is_unique

    @classmethod
    def from_dictionary(cls, arguments, entityset, dependencies, primitives_deserializer):
        base_feature = dependencies[arguments['base_feature']]
        relationship_path = [Relationship.from_dictionary(r, entityset)
                             for r in arguments['relationship_path']]
        child_entity = relationship_path[0].child_entity
        return cls(base_feature, child_entity, relationship_path=relationship_path)

    @property
    def variable(self):
        return self.base_features[0].variable

    @property
    def number_output_features(self):
        return self.base_features[0].primitive.number_output_features

    @property
    def default_value(self):
        return self.base_features[0].default_value

    def copy(self):
        """Return copy of feature"""
        return DirectFeature(self.base_features[0], self.entity,
                             relationship_path=self.relationship_path)

    @property
    def variable_type(self):
        return self.base_features[0].variable_type

    def generate_name(self):
        if self._path_is_unique:
            relationship_path_name = self.parent_entity.id
        else:
            relationship_names = [r.parent_name for r in self.relationship_path]
            relationship_path_name = '.'.join(relationship_names)
        return u"%s.%s" % (relationship_path_name,
                           self.base_features[0].get_name())

    def get_feature_names(self):
        return [u"%s.%s" % (self.parent_entity.id, base_name)
                for base_name in self.base_features[0].get_feature_names()]

    def get_arguments(self):
        return {
            'base_feature': self.base_features[0].unique_name(),
            'relationship_path': [r.to_dictionary() for r in self.relationship_path]
        }


class AggregationFeature(FeatureBase):
    # Feature to condition this feature by in
    # computation (e.g. take the Count of products where the product_id is
    # "basketball".)
    where = None
    #: (str or :class:`.Timedelta`): Use only some amount of previous data from
    # each time point during calculation
    use_previous = None

    def __init__(self, base_features, parent_entity, primitive,
                 relationship_path=None, use_previous=None, where=None):
        if hasattr(base_features, '__iter__'):
            base_features = [_check_feature(bf) for bf in base_features]
            msg = "all base features must share the same entity"
            assert len(set([bf.entity for bf in base_features])) == 1, msg
        else:
            base_features = [_check_feature(base_features)]

        self.child_entity = base_features[0].entity

        relationship_path, self._path_is_unique = \
            self._handle_relationship_path(parent_entity, relationship_path)

        self.parent_entity = parent_entity.entityset.metadata[parent_entity.id]

        if where is not None:
            self.where = _check_feature(where)
            msg = "Where feature must be defined on child entity {}".format(
                self.child_entity.id)
            assert self.where.entity.id == self.child_entity.id, msg

        if use_previous:
            assert self.child_entity.time_index is not None, (
                "Applying function that requires time index to entity that "
                "doesn't have one")

            self.use_previous = _check_timedelta(use_previous)
            assert len(base_features) > 0
            time_index = base_features[0].entity.time_index
            time_col = base_features[0].entity[time_index]
            assert time_index is not None, ("Use previous can only be defined "
                                            "on entities with a time index")
            assert _check_time_against_column(self.use_previous, time_col)

        super(AggregationFeature, self).__init__(entity=parent_entity,
                                                 base_features=base_features,
                                                 relationship_path=relationship_path,
                                                 primitive=primitive)

    def _handle_relationship_path(self, parent_entity, relationship_path):
        if relationship_path:
            first_parent = relationship_path[0].parent_entity
            assert parent_entity == first_parent, \
                'parent_entity must match first relationship in path.'

            assert self.child_entity == relationship_path[-1].child_entity, \
                'Base feature must be defined on the entity at the end of relationship_path'

            path_is_unique = parent_entity.entityset \
                .has_unique_forward_path(self.child_entity.id, parent_entity.id)
        else:
            relationship_path = _find_path(parent_entity.id,
                                           self.child_entity.id,
                                           parent_entity.entityset,
                                           backward=True)
            path_is_unique = True

        return relationship_path, path_is_unique

    @classmethod
    def from_dictionary(cls, arguments, entityset, dependencies, primitives_deserializer):
        base_features = [dependencies[name] for name in arguments['base_features']]
        relationship_path = [Relationship.from_dictionary(r, entityset)
                             for r in arguments['relationship_path']]
        parent_entity = relationship_path[0].parent_entity

        primitive = primitives_deserializer.deserialize_primitive(arguments['primitive'])

        use_previous_data = arguments['use_previous']
        use_previous = use_previous_data and Timedelta.from_dictionary(use_previous_data)

        where_name = arguments['where']
        where = where_name and dependencies[where_name]

        return cls(base_features, parent_entity, primitive, relationship_path=relationship_path,
                   use_previous=use_previous, where=where)

    def copy(self):
        return AggregationFeature(self.base_features,
                                  parent_entity=self.parent_entity,
                                  relationship_path=self.relationship_path,
                                  primitive=self.primitive,
                                  use_previous=self.use_previous,
                                  where=self.where)

    def _where_str(self):
        if self.where is not None:
            where_str = u" WHERE " + self.where.get_name()
        else:
            where_str = ''
        return where_str

    def _use_prev_str(self):
        if self.use_previous is not None:
            use_prev_str = u", Last {}".format(self.use_previous.get_name())
        else:
            use_prev_str = u''
        return use_prev_str

    def generate_name(self):
        if self._path_is_unique:
            relationship_path_name = self.child_entity.id
        else:
            relationship_names = [r.child_name for r in self.relationship_path]
            relationship_path_name = '.'.join(relationship_names)
        return self.primitive.generate_name(base_feature_names=[bf.get_name() for bf in self.base_features],
                                            relationship_path_name=relationship_path_name,
                                            parent_entity_id=self.parent_entity.id,
                                            where_str=self._where_str(),
                                            use_prev_str=self._use_prev_str())

    def get_arguments(self):
        return {
            'base_features': [feat.unique_name() for feat in self.base_features],
            'relationship_path': [r.to_dictionary() for r in self.relationship_path],
            'primitive': serialize_primitive(self.primitive),
            'where': self.where and self.where.unique_name(),
            'use_previous': self.use_previous and self.use_previous.get_arguments(),
        }


class TransformFeature(FeatureBase):
    def __init__(self, base_features, primitive):
        # Any edits made to this method should also be made to the
        # new_class_init method in make_trans_primitive
        if hasattr(base_features, '__iter__'):
            base_features = [_check_feature(bf) for bf in base_features]
            msg = "all base features must share the same entity"
            assert len(set([bf.entity for bf in base_features])) == 1, msg
        else:
            base_features = [_check_feature(base_features)]

        # R TODO handle stacking on sub-features
        assert all(bf.number_output_features == 1 for bf in base_features)

        super(TransformFeature, self).__init__(entity=base_features[0].entity,
                                               base_features=base_features,
                                               relationship_path=[],
                                               primitive=primitive)

    @classmethod
    def from_dictionary(cls, arguments, entityset, dependencies, primitives_deserializer):
        base_features = [dependencies[name] for name in arguments['base_features']]
        primitive = primitives_deserializer.deserialize_primitive(arguments['primitive'])
        return cls(base_features, primitive)

    def copy(self):
        return TransformFeature(self.base_features, self.primitive)

    def generate_name(self):
        return self.primitive.generate_name(base_feature_names=[bf.get_name() for bf in self.base_features])

    def get_arguments(self):
        return {
            'base_features': [feat.unique_name() for feat in self.base_features],
            'primitive': serialize_primitive(self.primitive)
        }


class GroupByTransformFeature(TransformFeature):
    def __init__(self, base_features, primitive, groupby):
        if not isinstance(groupby, FeatureBase):
            groupby = IdentityFeature(groupby)
        assert issubclass(groupby.variable_type, Discrete)
        self.groupby = groupby

        if hasattr(base_features, '__iter__'):
            base_features.append(groupby)
        else:
            base_features = [base_features, groupby]

        super(GroupByTransformFeature, self).__init__(base_features=base_features,
                                                      primitive=primitive)

    @classmethod
    def from_dictionary(cls, arguments, entityset, dependencies, primitives_deserializer):
        base_features = [dependencies[name] for name in arguments['base_features']]
        primitive = primitives_deserializer.deserialize_primitive(arguments['primitive'])
        groupby = dependencies[arguments['groupby']]
        return cls(base_features, primitive, groupby)

    def copy(self):
        # the groupby feature is appended to base_features in the __init__
        # so here we separate them again
        return GroupByTransformFeature(self.base_features[:-1],
                                       self.primitive,
                                       self.groupby)

    def generate_name(self):
        # exclude the groupby feature from base_names since it has a special
        # place in the feature name
        base_names = [bf.get_name() for bf in self.base_features[:-1]]
        _name = self.primitive.generate_name(base_names)
        return u"{} by {}".format(_name, self.groupby.get_name())

    def get_arguments(self):
        # Do not include groupby in base_features.
        feature_names = [feat.unique_name() for feat in self.base_features
                         if feat.unique_name() != self.groupby.unique_name()]
        return {
            'base_features': feature_names,
            'primitive': serialize_primitive(self.primitive),
            'groupby': self.groupby.unique_name(),
        }


class Feature(object):
    """
    Alias to create feature. Infers the feature type based on init parameters.
    """

    def __new__(self, base, entity=None, groupby=None, parent_entity=None,
                primitive=None, use_previous=None, where=None):

        # either direct or indentity
        if primitive is None and entity is None:
            return IdentityFeature(base)
        elif primitive is None and entity is not None:
            return DirectFeature(base, entity)
        elif primitive is not None and parent_entity is not None:
            assert isinstance(primitive, AggregationPrimitive) or issubclass(primitive, AggregationPrimitive)
            return AggregationFeature(base, parent_entity=parent_entity,
                                      use_previous=use_previous, where=where,
                                      primitive=primitive)
        elif primitive is not None:
            assert (isinstance(primitive, TransformPrimitive) or
                    issubclass(primitive, TransformPrimitive))
            if groupby is not None:
                return GroupByTransformFeature(base,
                                               primitive=primitive,
                                               groupby=groupby)
            return TransformFeature(base, primitive=primitive)

        raise Exception("Unrecognized feature initialization")


def _check_feature(feature):
    if isinstance(feature, Variable):
        return IdentityFeature(feature)
    elif isinstance(feature, FeatureBase):
        return feature

    raise Exception("Not a feature")


def _find_path(start_entity_id, end_entity_id, es, backward=False):
    """
    Finds a path of relationships between start and end.
    Raises if there is no path or multiple possible paths.
    """
    if backward:
        paths = es.find_backward_paths(start_entity_id, end_entity_id)
    else:
        paths = es.find_forward_paths(start_entity_id, end_entity_id)

    path = next(paths, None)

    if not path:
        raise RuntimeError('No path from "%s" to "%s" found.'
                           % (start_entity_id, end_entity_id))
    # Check for another path.
    elif next(paths, None):
        message = "There are multiple possible paths to the base entity. " \
                  "You must specify a relationship path."
        raise RuntimeError(message)
    else:
        return path
