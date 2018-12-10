import copy
import functools

from .primitive_base import PrimitiveBase
from .utils import inspect_function_args


class AggregationPrimitive(PrimitiveBase):
    """Feature for a parent entity that summarizes
        related instances in a child entity"""
    stack_on = None  # whitelist of primitives that can be in input_types
    stack_on_exclude = None  # blacklist of primitives that can be insigniture
    base_of = None  # whitelist of primitives this prim can be input for
    base_of_exclude = None  # primitives this primitive can't be input for
    stack_on_self = True  # whether or not it can be in input_types of self
    allow_where = True  # whether DFS can apply where clause to this primitive

    def __init__(self, base_features, parent_entity, use_previous=None,
                 where=None):
        # Any edits made to this method should also be made to the
        # new_class_init method in make_agg_primitive
        if not hasattr(base_features, '__iter__'):
            base_features = [self._check_feature(base_features)]
        else:
            base_features = [self._check_feature(bf) for bf in base_features]
            msg = "all base features must share the same entity"
            assert len(set([bf.entity for bf in base_features])) == 1, msg
        self.base_features = base_features[:]

        self.child_entity = base_features[0].entity

        if where is not None:
            self.where = self._check_feature(where)
            msg = "Where feature must be defined on child entity {}".format(
                self.child_entity.id)
            assert self.where.entity.id == self.child_entity.id, msg

        if use_previous:
            assert self.child_entity.time_index is not None, (
                "Applying function that requires time index to entity that "
                "doesn't have one")

        self.use_previous = use_previous

        super(AggregationPrimitive, self).__init__(parent_entity,
                                                   self.base_features)

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

    def _base_feature_str(self):
        return u', ' \
            .join([bf.get_name() for bf in self.base_features])

    def generate_name(self):
        where_str = self._where_str()
        use_prev_str = self._use_prev_str()

        base_features_str = self._base_feature_str()

        return u"%s(%s.%s%s%s)" % (self.name.upper(),
                                   self.child_entity.id,
                                   base_features_str,
                                   where_str, use_prev_str)


def make_agg_primitive(function, input_types, return_type, name=None,
                       stack_on_self=True, stack_on=None,
                       stack_on_exclude=None, base_of=None,
                       base_of_exclude=None, description='A custom primitive',
                       cls_attributes=None, uses_calc_time=False,
                       commutative=False):
    '''Returns a new aggregation primitive class. The primitive infers default
    values by passing in empty data.

    Args:
        function (function): Function that takes in an array  and applies some
            transformation to it.

        input_types (list[Variable]): Variable types of the inputs.

        return_type (Variable): Variable type of return.

        name (str): Name of the function.  If no name is provided, the name
            of `function` will be used.

        stack_on_self (bool): Whether this primitive can be in input_types of self.

        stack_on (list[PrimitiveBase]): Whitelist of primitives that
            can be input_types.

        stack_on_exclude (list[PrimitiveBase]): Blacklist of
            primitives that cannot be input_types.

        base_of (list[PrimitiveBase): Whitelist of primitives that
            can have this primitive in input_types.

        base_of_exclude (list[PrimitiveBase]): Blacklist of
            primitives that cannot have this primitive in input_types.

        description (str): Description of primitive.

        cls_attributes (dict[str -> anytype]): Custom attributes to be added to                     class. Key is attribute name, value is the attribute value.

        uses_calc_time (bool): If True, the cutoff time the feature is being
            calculated at will be passed to the function as the keyword
            argument 'time'.

        commutative (bool): If True, will only make one feature per unique set
            of base features.

    Example:
        .. ipython :: python

            from featuretools.primitives import make_agg_primitive
            from featuretools.variable_types import DatetimeTimeIndex, Numeric

            def time_since_last(values, time=None):
                time_since = time - values.iloc[0]
                return time_since.total_seconds()

            TimeSinceLast = make_agg_primitive(
                function=time_since_last,
                input_types=[DatetimeTimeIndex],
                return_type=Numeric,
                description="Time since last related instance",
                uses_calc_time=True)

    '''
    cls = {"__doc__": description}
    if cls_attributes is not None:
        cls.update(cls_attributes)
    name = name or function.__name__
    new_class = type(name, (AggregationPrimitive,), cls)
    new_class.name = name
    new_class.input_types = input_types
    new_class.return_type = return_type
    new_class.stack_on = stack_on
    new_class.stack_on_exclude = stack_on_exclude
    new_class.stack_on_self = stack_on_self
    new_class.base_of = base_of
    new_class.base_of_exclude = base_of_exclude
    new_class.commutative = commutative
    new_class, default_kwargs = inspect_function_args(new_class,
                                                      function,
                                                      uses_calc_time)

    if len(default_kwargs) > 0:
        new_class.default_kwargs = default_kwargs

        def new_class_init(self, base_features, parent_entity,
                           use_previous=None, where=None, **kwargs):
            if not hasattr(base_features, '__iter__'):
                base_features = [self._check_feature(base_features)]
            else:
                base_features = [self._check_feature(bf)
                                 for bf in base_features]
                msg = "all base features must share the same entity"
                assert len(set([bf.entity for bf in base_features])) == 1, msg
            self.base_features = base_features[:]

            self.child_entity = base_features[0].entity

            if where is not None:
                self.where = self._check_feature(where)
                msg = "Where feature must be defined on child entity {}"
                msg = msg.format(self.child_entity.id)
                assert self.where.entity.id == self.child_entity.id, msg

            if use_previous:
                assert self.child_entity.time_index is not None, (
                    "Applying function that requires time index to entity that"
                    " doesn't have one")

            self.use_previous = use_previous
            self.kwargs = copy.deepcopy(self.default_kwargs)
            self.kwargs.update(kwargs)
            self.partial = functools.partial(function, **self.kwargs)
            self.partial.__name__ = name

            super(AggregationPrimitive, self).__init__(parent_entity,
                                                       self.base_features)
        new_class.__init__ = new_class_init
        new_class.get_function = lambda self: self.partial
    else:
        # creates a lambda function that returns function every time
        new_class.get_function = lambda self, f=function: f

    # infers default_value by passing empty data
    try:
        new_class.default_value = function(*[[]] * len(input_types))
    except Exception:
        pass

    return new_class
