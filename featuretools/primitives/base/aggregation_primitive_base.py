import copy
import functools
import inspect

from .primitive_base import PrimitiveBase
from .utils import inspect_function_args


class AggregationPrimitive(PrimitiveBase):
    stack_on = None  # whitelist of primitives that can be in input_types
    stack_on_exclude = None  # blacklist of primitives that can be insigniture
    base_of = None  # whitelist of primitives this prim can be input for
    base_of_exclude = None  # primitives this primitive can't be input for
    stack_on_self = True  # whether or not it can be in input_types of self
    allow_where = True  # whether DFS can apply where clause to this primitive

    def generate_name(self, base_feature_names, child_entity_id,
                      parent_entity_id, where_str, use_prev_str):
        base_features_str = ", ".join(base_feature_names)
        return u"%s(%s.%s%s%s%s)" % (
            self.name.upper(),
            child_entity_id,
            base_features_str,
            where_str,
            use_prev_str,
            self.get_args_string(),
        )


def make_agg_primitive(function, input_types, return_type, name=None,
                       stack_on_self=True, stack_on=None,
                       stack_on_exclude=None, base_of=None,
                       base_of_exclude=None, description=None,
                       cls_attributes=None, uses_calc_time=False,
                       default_value=None, commutative=False,
                       number_output_features=1):
    '''Returns a new aggregation primitive class. The primitive infers default
    values by passing in empty data.

    Args:
        function (function): Function that takes in a series and applies some
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

        cls_attributes (dict[str -> anytype]): Custom attributes to be added to
            class. Key is attribute name, value is the attribute value.

        uses_calc_time (bool): If True, the cutoff time the feature is being
            calculated at will be passed to the function as the keyword
            argument 'time'.

        default_value (Variable): Default value when creating the primitive to
            avoid the inference step. If no default value if provided, the
            inference happen.

        commutative (bool): If True, will only make one feature per unique set
            of base features.

        number_output_features (int): The number of output features (columns in
            the matrix) associated with this feature.

    Example:
        .. ipython :: python

            from featuretools.primitives import make_agg_primitive
            from featuretools.variable_types import DatetimeTimeIndex, Numeric

            def time_since_last(values, time=None):
                time_since = time - values.iloc[-1]
                return time_since.total_seconds()

            TimeSinceLast = make_agg_primitive(
                function=time_since_last,
                input_types=[DatetimeTimeIndex],
                return_type=Numeric,
                description="Time since last related instance",
                uses_calc_time=True)

    '''
    if description is None:
        default_description = 'A custom primitive'
        doc = inspect.getdoc(function)
        description = doc if doc is not None else default_description
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
    new_class.number_output_features = number_output_features
    new_class, default_kwargs = inspect_function_args(new_class,
                                                      function,
                                                      uses_calc_time)

    if len(default_kwargs) > 0:
        new_class.default_kwargs = default_kwargs

        def new_class_init(self, **kwargs):
            self.kwargs = copy.deepcopy(self.default_kwargs)
            self.kwargs.update(kwargs)
            self.partial = functools.partial(function, **self.kwargs)
            self.partial.__name__ = name

        new_class.__init__ = new_class_init
        new_class.get_function = lambda self: self.partial
    else:
        # creates a lambda function that returns function every time
        new_class.get_function = lambda self, f=function: f

    if default_value is None:
        # infers default_value by passing empty data
        try:
            new_class.default_value = function(*[[]] * len(input_types))
        except Exception:
            pass
    else:
        # avoiding the inference step
        new_class.default_value = default_value

    return new_class
