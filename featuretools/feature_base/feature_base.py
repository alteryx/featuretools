from woodwork.column_schema import ColumnSchema
from woodwork.logical_types import Boolean, BooleanNullable

from featuretools import primitives
from featuretools.entityset.relationship import Relationship, RelationshipPath
from featuretools.entityset.timedelta import Timedelta
from featuretools.feature_base.utils import is_valid_input
from featuretools.primitives.base import (
    AggregationPrimitive,
    PrimitiveBase,
    TransformPrimitive,
)
from featuretools.utils.gen_utils import Library, import_or_none, is_instance
from featuretools.utils.wrangle import _check_time_against_column, _check_timedelta

dd = import_or_none("dask.dataframe")
ps = import_or_none("pyspark.pandas")

_ES_REF = {}


class FeatureBase(object):
    def __init__(
        self,
        dataframe,
        base_features,
        relationship_path,
        primitive,
        name=None,
        names=None,
    ):
        """Base class for all features

        Args:
            entityset (EntitySet): entityset this feature is being calculated for
            dataframe (DataFrame): dataframe for calculating this feature
            base_features (list[FeatureBase]): list of base features for primitive
            relationship_path (RelationshipPath): path from this dataframe to the
                dataframe of the base features.
            primitive (:class:`.PrimitiveBase`): primitive to calculate. if not initialized when passed, gets initialized with no arguments
        """
        assert all(
            isinstance(f, FeatureBase) for f in base_features
        ), "All base features must be features"

        self.dataframe_name = dataframe.ww.name
        self.entityset = _ES_REF[dataframe.ww.metadata["entityset_id"]]

        self.base_features = base_features

        # initialize if not already initialized
        if not isinstance(primitive, PrimitiveBase):
            primitive = primitive()

        # default library is PANDAS
        if is_instance(dataframe, dd, "DataFrame"):
            primitive.series_library = Library.DASK
        elif is_instance(dataframe, ps, "DataFrame"):
            primitive.series_library = Library.SPARK

        self.primitive = primitive

        self.relationship_path = relationship_path

        self._name = name

        self._names = names

        assert self._check_input_types(), (
            "Provided inputs don't match input " "type requirements"
        )

    def __getitem__(self, key):
        assert (
            self.number_output_features > 1
        ), "can only access slice of multi-output feature"
        assert (
            self.number_output_features > key
        ), "index is higher than the number of outputs"
        return FeatureOutputSlice(self, key)

    @classmethod
    def from_dictionary(cls, arguments, entityset, dependencies, primitive):
        raise NotImplementedError("Must define from_dictionary on FeatureBase subclass")

    def rename(self, name):
        """Rename Feature, returns copy. Will reset any custom feature column names
        to their default value."""
        feature_copy = self.copy()
        feature_copy._name = name
        feature_copy._names = None
        return feature_copy

    def copy(self):
        raise NotImplementedError("Must define copy on FeatureBase subclass")

    def get_name(self):
        if not self._name:
            self._name = self.generate_name()
        return self._name

    def get_feature_names(self):
        if not self._names:
            if self.number_output_features == 1:
                self._names = [self.get_name()]
            else:
                self._names = self.generate_names()
                if self.get_name() != self.generate_name():
                    self._names = [
                        self.get_name() + "[{}]".format(i)
                        for i in range(len(self._names))
                    ]
        return self._names

    def set_feature_names(self, names):
        """Set new values for the feature column names, overriding the default values.
        Number of names provided must match the number of output columns defined for
        the feature, and all provided names should be unique. Only works for features
        that have more than one output column. Use ``Feature.rename`` to change the column
        name for single output features.

        Args:
            names (list[str]): List of names to use for the output feature columns. Provided
                names must be unique.
        """
        if self.number_output_features == 1:
            raise ValueError(
                "The set_feature_names can only be used on features that have more than one output column.",
            )

        num_new_names = len(names)
        if self.number_output_features != num_new_names:
            raise ValueError(
                "Number of names provided must match the number of output features:"
                f" {num_new_names} name(s) provided, {self.number_output_features} expected.",
            )

        if len(set(names)) != num_new_names:
            raise ValueError("Provided output feature names must be unique.")

        self._names = names

    def get_function(self, **kwargs):
        return self.primitive.get_function(**kwargs)

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
        deps = [d for d in deps if d.unique_name() not in ignored]

        if deep:
            for dep in deps[:]:  # copy so we don't modify list we iterate over
                deep_deps = dep.get_dependencies(deep, ignored)
                deps += deep_deps

        return deps

    def get_depth(self, stop_at=None):
        """Returns depth of feature"""
        max_depth = 0
        stop_at_set = set()
        if stop_at is not None:
            stop_at_set = set([i.unique_name() for i in stop_at])
            if self.unique_name() in stop_at_set:
                return 0
        for dep in self.get_dependencies(deep=True, ignored=stop_at_set):
            max_depth = max(dep.get_depth(stop_at=stop_at), max_depth)
        return max_depth + 1

    def _check_input_types(self):
        if len(self.base_features) == 0:
            return True

        input_types = self.primitive.input_types
        if input_types is not None:
            if not isinstance(input_types[0], list):
                input_types = [input_types]

            for t in input_types:
                zipped = list(zip(t, self.base_features))
                if all([is_valid_input(f.column_schema, t) for t, f in zipped]):
                    return True
        else:
            return True
        return False

    @property
    def dataframe(self):
        """Dataframe this feature belongs too"""
        return self.entityset[self.dataframe_name]

    @property
    def number_output_features(self):
        return self.primitive.number_output_features

    def __repr__(self):
        return "<Feature: %s>" % (self.get_name())

    def hash(self):
        return hash(self.get_name() + self.dataframe_name)

    def __hash__(self):
        return self.hash()

    @property
    def column_schema(self):
        feature = self
        column_schema = self.primitive.return_type

        while column_schema is None:
            # get column_schema of first base feature
            base_feature = feature.base_features[0]
            column_schema = base_feature.column_schema

            # only the original time index should exist
            # so make this feature's return type just a Datetime
            if "time_index" in column_schema.semantic_tags:
                column_schema = ColumnSchema(
                    logical_type=column_schema.logical_type,
                    semantic_tags=column_schema.semantic_tags - {"time_index"},
                )
            elif "index" in column_schema.semantic_tags:
                column_schema = ColumnSchema(
                    logical_type=column_schema.logical_type,
                    semantic_tags=column_schema.semantic_tags - {"index"},
                )
                # Need to add back in the numeric standard tag so the schema can get recognized
                # as a valid return type
                if column_schema.is_numeric:
                    column_schema.semantic_tags.add("numeric")
                if column_schema.is_categorical:
                    column_schema.semantic_tags.add("category")

            # direct features should keep the foreign key tag, but all other features should get converted
            if (
                not isinstance(feature, DirectFeature)
                and "foreign_key" in column_schema.semantic_tags
            ):
                column_schema = ColumnSchema(
                    logical_type=column_schema.logical_type,
                    semantic_tags=column_schema.semantic_tags - {"foreign_key"},
                )

            feature = base_feature

        return column_schema

    @property
    def default_value(self):
        return self.primitive.default_value

    def get_arguments(self):
        raise NotImplementedError("Must define get_arguments on FeatureBase subclass")

    def to_dictionary(self):
        return {
            "type": type(self).__name__,
            "dependencies": [dep.unique_name() for dep in self.get_dependencies()],
            "arguments": self.get_arguments(),
        }

    def _handle_binary_comparison(self, other, Primitive, PrimitiveScalar):
        if isinstance(other, FeatureBase):
            return Feature([self, other], primitive=Primitive)

        return Feature([self], primitive=PrimitiveScalar(other))

    def __eq__(self, other):
        """Compares to other by equality"""
        return self._handle_binary_comparison(
            other,
            primitives.Equal,
            primitives.EqualScalar,
        )

    def __ne__(self, other):
        """Compares to other by non-equality"""
        return self._handle_binary_comparison(
            other,
            primitives.NotEqual,
            primitives.NotEqualScalar,
        )

    def __gt__(self, other):
        """Compares if greater than other"""
        return self._handle_binary_comparison(
            other,
            primitives.GreaterThan,
            primitives.GreaterThanScalar,
        )

    def __ge__(self, other):
        """Compares if greater than or equal to other"""
        return self._handle_binary_comparison(
            other,
            primitives.GreaterThanEqualTo,
            primitives.GreaterThanEqualToScalar,
        )

    def __lt__(self, other):
        """Compares if less than other"""
        return self._handle_binary_comparison(
            other,
            primitives.LessThan,
            primitives.LessThanScalar,
        )

    def __le__(self, other):
        """Compares if less than or equal to other"""
        return self._handle_binary_comparison(
            other,
            primitives.LessThanEqualTo,
            primitives.LessThanEqualToScalar,
        )

    def __add__(self, other):
        """Add other"""
        return self._handle_binary_comparison(
            other,
            primitives.AddNumeric,
            primitives.AddNumericScalar,
        )

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        """Subtract other"""
        return self._handle_binary_comparison(
            other,
            primitives.SubtractNumeric,
            primitives.SubtractNumericScalar,
        )

    def __rsub__(self, other):
        return Feature([self], primitive=primitives.ScalarSubtractNumericFeature(other))

    def __div__(self, other):
        """Divide by other"""
        return self._handle_binary_comparison(
            other,
            primitives.DivideNumeric,
            primitives.DivideNumericScalar,
        )

    def __truediv__(self, other):
        return self.__div__(other)

    def __rtruediv__(self, other):
        return self.__rdiv__(other)

    def __rdiv__(self, other):
        return Feature([self], primitive=primitives.DivideByFeature(other))

    def __mul__(self, other):
        """Multiply by other"""
        if isinstance(other, FeatureBase):
            if all(
                [
                    isinstance(f.column_schema.logical_type, (Boolean, BooleanNullable))
                    for f in (self, other)
                ],
            ):
                return Feature([self, other], primitive=primitives.MultiplyBoolean)
            if (
                "numeric" in self.column_schema.semantic_tags
                and isinstance(
                    other.column_schema.logical_type,
                    (Boolean, BooleanNullable),
                )
                or "numeric" in other.column_schema.semantic_tags
                and isinstance(
                    self.column_schema.logical_type,
                    (Boolean, BooleanNullable),
                )
            ):
                return Feature(
                    [self, other],
                    primitive=primitives.MultiplyNumericBoolean,
                )
        return self._handle_binary_comparison(
            other,
            primitives.MultiplyNumeric,
            primitives.MultiplyNumericScalar,
        )

    def __rmul__(self, other):
        return self.__mul__(other)

    def __mod__(self, other):
        """Take modulus of other"""
        return self._handle_binary_comparison(
            other,
            primitives.ModuloNumeric,
            primitives.ModuloNumericScalar,
        )

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
        return Feature(
            [self],
            primitive=primitives.IsIn(list_of_outputs=list_of_output),
        )

    def is_null(self):
        """Compares feature to null by equality"""
        return Feature([self], primitive=primitives.IsNull)

    def __invert__(self):
        return self.NOT()

    def unique_name(self):
        return "%s: %s" % (self.dataframe_name, self.get_name())

    def relationship_path_name(self):
        return self.relationship_path.name


class IdentityFeature(FeatureBase):
    """Feature for dataframe that is equivalent to underlying column"""

    def __init__(self, column, name=None):
        self.column_name = column.ww.name
        self.return_type = column.ww.schema

        metadata = column.ww.schema._metadata
        es = _ES_REF[metadata["entityset_id"]]
        super(IdentityFeature, self).__init__(
            dataframe=es[metadata["dataframe_name"]],
            base_features=[],
            relationship_path=RelationshipPath([]),
            primitive=PrimitiveBase,
            name=name,
        )

    @classmethod
    def from_dictionary(cls, arguments, entityset, dependencies, primitive):
        dataframe_name = arguments["dataframe_name"]
        column_name = arguments["column_name"]
        column = entityset[dataframe_name].ww[column_name]
        return cls(column=column, name=arguments["name"])

    def copy(self):
        """Return copy of feature"""
        return IdentityFeature(self.entityset[self.dataframe_name].ww[self.column_name])

    def generate_name(self):
        return self.column_name

    def get_depth(self, stop_at=None):
        return 0

    def get_arguments(self):
        return {
            "name": self.get_name(),
            "column_name": self.column_name,
            "dataframe_name": self.dataframe_name,
        }

    @property
    def column_schema(self):
        return self.return_type


class DirectFeature(FeatureBase):
    """Feature for child dataframe that inherits
    a feature value from a parent dataframe"""

    input_types = [ColumnSchema()]
    return_type = None

    def __init__(
        self,
        base_feature,
        child_dataframe_name,
        relationship=None,
        name=None,
    ):
        base_feature = _validate_base_features(base_feature)[0]
        self.parent_dataframe_name = base_feature.dataframe_name
        relationship = self._handle_relationship(
            base_feature.entityset,
            child_dataframe_name,
            relationship,
        )
        child_dataframe = base_feature.entityset[child_dataframe_name]
        super(DirectFeature, self).__init__(
            dataframe=child_dataframe,
            base_features=[base_feature],
            relationship_path=RelationshipPath([(True, relationship)]),
            primitive=PrimitiveBase,
            name=name,
        )

    def _handle_relationship(self, entityset, child_dataframe_name, relationship):
        child_dataframe = entityset[child_dataframe_name]
        if relationship:
            relationship_child = relationship.child_dataframe
            assert (
                child_dataframe.ww.name == relationship_child.ww.name
            ), "child_dataframe must be the relationship child dataframe"

            assert (
                self.parent_dataframe_name == relationship.parent_dataframe.ww.name
            ), "Base feature must be defined on the relationship parent dataframe"
        else:
            child_relationships = entityset.get_forward_relationships(
                child_dataframe.ww.name,
            )
            possible_relationships = (
                r
                for r in child_relationships
                if r.parent_dataframe.ww.name == self.parent_dataframe_name
            )
            relationship = next(possible_relationships, None)

            if not relationship:
                raise RuntimeError(
                    'No relationship from "%s" to "%s" found.'
                    % (child_dataframe.ww.name, self.parent_dataframe_name),
                )

            # Check for another path.
            elif next(possible_relationships, None):
                message = (
                    "There are multiple relationships to the base dataframe. "
                    "You must specify a relationship."
                )
                raise RuntimeError(message)

        return relationship

    @classmethod
    def from_dictionary(cls, arguments, entityset, dependencies, primitive):
        base_feature = dependencies[arguments["base_feature"]]
        relationship = Relationship.from_dictionary(
            arguments["relationship"],
            entityset,
        )
        child_dataframe_name = relationship.child_dataframe.ww.name
        return cls(
            base_feature=base_feature,
            child_dataframe_name=child_dataframe_name,
            relationship=relationship,
            name=arguments["name"],
        )

    @property
    def number_output_features(self):
        return self.base_features[0].number_output_features

    @property
    def default_value(self):
        return self.base_features[0].default_value

    def copy(self):
        """Return copy of feature"""
        _is_forward, relationship = self.relationship_path[0]
        return DirectFeature(
            self.base_features[0],
            self.dataframe_name,
            relationship=relationship,
        )

    @property
    def column_schema(self):
        return self.base_features[0].column_schema

    def generate_name(self):
        return self._name_from_base(self.base_features[0].get_name())

    def generate_names(self):
        return [
            self._name_from_base(base_name)
            for base_name in self.base_features[0].get_feature_names()
        ]

    def get_arguments(self):
        _is_forward, relationship = self.relationship_path[0]
        return {
            "name": self.get_name(),
            "base_feature": self.base_features[0].unique_name(),
            "relationship": relationship.to_dictionary(),
        }

    def _name_from_base(self, base_name):
        return "%s.%s" % (self.relationship_path_name(), base_name)


class AggregationFeature(FeatureBase):
    # Feature to condition this feature by in
    # computation (e.g. take the Count of products where the product_id is
    # "basketball".)
    where = None
    #: (str or :class:`.Timedelta`): Use only some amount of previous data from
    # each time point during calculation
    use_previous = None

    def __init__(
        self,
        base_features,
        parent_dataframe_name,
        primitive,
        relationship_path=None,
        use_previous=None,
        where=None,
        name=None,
    ):
        base_features = _validate_base_features(base_features)

        for bf in base_features:
            if bf.number_output_features > 1:
                raise ValueError("Cannot stack on whole multi-output feature.")

        self.child_dataframe_name = base_features[0].dataframe_name
        entityset = base_features[0].entityset
        relationship_path, self._path_is_unique = self._handle_relationship_path(
            entityset,
            parent_dataframe_name,
            relationship_path,
        )

        self.parent_dataframe_name = parent_dataframe_name

        if where is not None:
            self.where = _validate_base_features(where)[0]
            msg = "Where feature must be defined on child dataframe {}".format(
                self.child_dataframe_name,
            )
            assert self.where.dataframe_name == self.child_dataframe_name, msg

        if use_previous:
            assert entityset[self.child_dataframe_name].ww.time_index is not None, (
                "Applying function that requires time index to dataframe that "
                "doesn't have one"
            )
            self.use_previous = _check_timedelta(use_previous)
            assert len(base_features) > 0
            time_index = base_features[0].dataframe.ww.time_index
            time_col = base_features[0].dataframe.ww[time_index]
            assert time_index is not None, (
                "Use previous can only be defined " "on dataframes with a time index"
            )
            assert _check_time_against_column(self.use_previous, time_col)

        super(AggregationFeature, self).__init__(
            dataframe=entityset[parent_dataframe_name],
            base_features=base_features,
            relationship_path=relationship_path,
            primitive=primitive,
            name=name,
        )

    def _handle_relationship_path(
        self,
        entityset,
        parent_dataframe_name,
        relationship_path,
    ):
        parent_dataframe = entityset[parent_dataframe_name]
        child_dataframe = entityset[self.child_dataframe_name]

        if relationship_path:
            assert all(
                not is_forward for is_forward, _r in relationship_path
            ), "All relationships in path must be backward"

            _is_forward, first_relationship = relationship_path[0]
            first_parent = first_relationship.parent_dataframe
            assert (
                parent_dataframe.ww.name == first_parent.ww.name
            ), "parent_dataframe must match first relationship in path."

            _is_forward, last_relationship = relationship_path[-1]
            assert (
                child_dataframe.ww.name == last_relationship.child_dataframe.ww.name
            ), "Base feature must be defined on the dataframe at the end of relationship_path"

            path_is_unique = entityset.has_unique_forward_path(
                child_dataframe.ww.name,
                parent_dataframe.ww.name,
            )
        else:
            paths = entityset.find_backward_paths(
                parent_dataframe.ww.name,
                child_dataframe.ww.name,
            )
            first_path = next(paths, None)

            if not first_path:
                raise RuntimeError(
                    'No backward path from "%s" to "%s" found.'
                    % (parent_dataframe.ww.name, child_dataframe.ww.name),
                )
            # Check for another path.
            elif next(paths, None):
                message = (
                    "There are multiple possible paths to the base dataframe. "
                    "You must specify a relationship path."
                )
                raise RuntimeError(message)

            relationship_path = RelationshipPath([(False, r) for r in first_path])
            path_is_unique = True

        return relationship_path, path_is_unique

    @classmethod
    def from_dictionary(cls, arguments, entityset, dependencies, primitive):
        base_features = [dependencies[name] for name in arguments["base_features"]]
        relationship_path = [
            Relationship.from_dictionary(r, entityset)
            for r in arguments["relationship_path"]
        ]
        parent_dataframe_name = relationship_path[0].parent_dataframe.ww.name
        relationship_path = RelationshipPath([(False, r) for r in relationship_path])

        use_previous_data = arguments["use_previous"]
        use_previous = use_previous_data and Timedelta.from_dictionary(
            use_previous_data,
        )

        where_name = arguments["where"]
        where = where_name and dependencies[where_name]

        feat = cls(
            base_features=base_features,
            parent_dataframe_name=parent_dataframe_name,
            primitive=primitive,
            relationship_path=relationship_path,
            use_previous=use_previous,
            where=where,
            name=arguments["name"],
        )
        feat._names = arguments.get("feature_names")
        return feat

    def copy(self):
        return AggregationFeature(
            self.base_features,
            parent_dataframe_name=self.parent_dataframe_name,
            relationship_path=self.relationship_path,
            primitive=self.primitive,
            use_previous=self.use_previous,
            where=self.where,
        )

    def _where_str(self):
        if self.where is not None:
            where_str = " WHERE " + self.where.get_name()
        else:
            where_str = ""
        return where_str

    def _use_prev_str(self):
        if self.use_previous is not None and hasattr(self.use_previous, "get_name"):
            use_prev_str = ", Last {}".format(self.use_previous.get_name())
        else:
            use_prev_str = ""
        return use_prev_str

    def generate_name(self):
        return self.primitive.generate_name(
            base_feature_names=[bf.get_name() for bf in self.base_features],
            relationship_path_name=self.relationship_path_name(),
            parent_dataframe_name=self.parent_dataframe_name,
            where_str=self._where_str(),
            use_prev_str=self._use_prev_str(),
        )

    def generate_names(self):
        return self.primitive.generate_names(
            base_feature_names=[bf.get_name() for bf in self.base_features],
            relationship_path_name=self.relationship_path_name(),
            parent_dataframe_name=self.parent_dataframe_name,
            where_str=self._where_str(),
            use_prev_str=self._use_prev_str(),
        )

    def get_arguments(self):
        arg_dict = {
            "name": self.get_name(),
            "base_features": [feat.unique_name() for feat in self.base_features],
            "relationship_path": [r.to_dictionary() for _, r in self.relationship_path],
            "primitive": self.primitive,
            "where": self.where and self.where.unique_name(),
            "use_previous": self.use_previous and self.use_previous.get_arguments(),
        }
        if self.number_output_features > 1:
            arg_dict["feature_names"] = self.get_feature_names()
        return arg_dict

    def relationship_path_name(self):
        if self._path_is_unique:
            return self.child_dataframe_name
        else:
            return self.relationship_path.name


class TransformFeature(FeatureBase):
    def __init__(self, base_features, primitive, name=None):
        base_features = _validate_base_features(base_features)

        for bf in base_features:
            if bf.number_output_features > 1:
                raise ValueError("Cannot stack on whole multi-output feature.")
        dataframe = base_features[0].entityset[base_features[0].dataframe_name]
        super(TransformFeature, self).__init__(
            dataframe=dataframe,
            base_features=base_features,
            relationship_path=RelationshipPath([]),
            primitive=primitive,
            name=name,
        )

    @classmethod
    def from_dictionary(cls, arguments, entityset, dependencies, primitive):
        base_features = [dependencies[name] for name in arguments["base_features"]]
        feat = cls(
            base_features=base_features,
            primitive=primitive,
            name=arguments["name"],
        )
        feat._names = arguments.get("feature_names")
        return feat

    def copy(self):
        return TransformFeature(self.base_features, self.primitive)

    def generate_name(self):
        return self.primitive.generate_name(
            base_feature_names=[bf.get_name() for bf in self.base_features],
        )

    def generate_names(self):
        return self.primitive.generate_names(
            base_feature_names=[bf.get_name() for bf in self.base_features],
        )

    def get_arguments(self):
        arg_dict = {
            "name": self.get_name(),
            "base_features": [feat.unique_name() for feat in self.base_features],
            "primitive": self.primitive,
        }
        if self.number_output_features > 1:
            arg_dict["feature_names"] = self.get_feature_names()
        return arg_dict


class GroupByTransformFeature(TransformFeature):
    def __init__(self, base_features, primitive, groupby, name=None):
        if not isinstance(groupby, FeatureBase):
            groupby = IdentityFeature(groupby)
        assert (
            len({"category", "foreign_key"} - groupby.column_schema.semantic_tags) < 2
        )
        self.groupby = groupby

        base_features = _validate_base_features(base_features)
        base_features.append(groupby)

        super(GroupByTransformFeature, self).__init__(
            base_features=base_features,
            primitive=primitive,
            name=name,
        )

    @classmethod
    def from_dictionary(cls, arguments, entityset, dependencies, primitive):
        base_features = [dependencies[name] for name in arguments["base_features"]]
        groupby = dependencies[arguments["groupby"]]
        feat = cls(
            base_features=base_features,
            primitive=primitive,
            groupby=groupby,
            name=arguments["name"],
        )
        feat._names = arguments.get("feature_names")
        return feat

    def copy(self):
        # the groupby feature is appended to base_features in the __init__
        # so here we separate them again
        return GroupByTransformFeature(
            self.base_features[:-1],
            self.primitive,
            self.groupby,
        )

    def generate_name(self):
        # exclude the groupby feature from base_names since it has a special
        # place in the feature name
        base_names = [bf.get_name() for bf in self.base_features[:-1]]
        _name = self.primitive.generate_name(base_names)
        return "{} by {}".format(_name, self.groupby.get_name())

    def generate_names(self):
        base_names = [bf.get_name() for bf in self.base_features[:-1]]
        _names = self.primitive.generate_names(base_names)
        names = [name + " by {}".format(self.groupby.get_name()) for name in _names]
        return names

    def get_arguments(self):
        # Do not include groupby in base_features.
        feature_names = [
            feat.unique_name()
            for feat in self.base_features
            if feat.unique_name() != self.groupby.unique_name()
        ]
        arg_dict = {
            "name": self.get_name(),
            "base_features": feature_names,
            "primitive": self.primitive,
            "groupby": self.groupby.unique_name(),
        }
        if self.number_output_features > 1:
            arg_dict["feature_names"] = self.get_feature_names()
        return arg_dict


class Feature(object):
    """
    Alias to create feature. Infers the feature type based on init parameters.
    """

    def __new__(
        self,
        base,
        dataframe_name=None,
        groupby=None,
        parent_dataframe_name=None,
        primitive=None,
        use_previous=None,
        where=None,
    ):
        # either direct or identity
        if primitive is None and dataframe_name is None:
            return IdentityFeature(base)
        elif primitive is None and dataframe_name is not None:
            return DirectFeature(base, dataframe_name)
        elif primitive is not None and parent_dataframe_name is not None:
            assert isinstance(primitive, AggregationPrimitive) or issubclass(
                primitive,
                AggregationPrimitive,
            )
            return AggregationFeature(
                base,
                parent_dataframe_name=parent_dataframe_name,
                use_previous=use_previous,
                where=where,
                primitive=primitive,
            )
        elif primitive is not None:
            assert isinstance(primitive, TransformPrimitive) or issubclass(
                primitive,
                TransformPrimitive,
            )
            if groupby is not None:
                return GroupByTransformFeature(
                    base,
                    primitive=primitive,
                    groupby=groupby,
                )
            return TransformFeature(base, primitive=primitive)

        raise Exception("Unrecognized feature initialization")


class FeatureOutputSlice(FeatureBase):
    """
    Class to access specific multi output feature column
    """

    def __init__(self, base_feature, n, name=None):
        base_features = [base_feature]
        self.num_output_parent = base_feature.number_output_features

        msg = "cannot access slice from single output feature"
        assert self.num_output_parent > 1, msg
        msg = "cannot access column that is not between 0 and " + str(
            self.num_output_parent - 1,
        )
        assert n < self.num_output_parent, msg

        self.n = n
        self._name = name
        self._names = [name] if name else None
        self.base_features = base_features
        self.base_feature = base_features[0]

        self.dataframe_name = base_feature.dataframe_name
        self.entityset = base_feature.entityset
        self.primitive = base_feature.primitive

        self.relationship_path = base_feature.relationship_path

    def __getitem__(self, key):
        raise ValueError("Cannot get item from slice of multi output feature")

    def generate_name(self):
        return self.base_feature.get_feature_names()[self.n]

    @property
    def number_output_features(self):
        return 1

    def get_arguments(self):
        return {
            "name": self.get_name(),
            "base_feature": self.base_feature.unique_name(),
            "n": self.n,
        }

    @classmethod
    def from_dictionary(cls, arguments, entityset, dependencies, primitive):
        base_feature_name = arguments["base_feature"]
        base_feature = dependencies[base_feature_name]
        n = arguments["n"]
        name = arguments["name"]
        return cls(base_feature=base_feature, n=n, name=name)

    def copy(self):
        return FeatureOutputSlice(self.base_feature, self.n)


def _validate_base_features(feature):
    if "Series" == type(feature).__name__:
        return [IdentityFeature(feature)]
    elif hasattr(feature, "__iter__"):
        features = [_validate_base_features(f)[0] for f in feature]
        msg = "all base features must share the same dataframe"
        assert len(set([bf.dataframe_name for bf in features])) == 1, msg
        return features
    elif isinstance(feature, FeatureBase):
        return [feature]
    else:
        raise Exception("Not a feature")
