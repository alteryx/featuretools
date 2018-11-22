import copy
import functools

from .primitive_base import PrimitiveBase
from .utils import inspect_function_args


class TransformPrimitive(PrimitiveBase):
    """Feature for entity that is a based off one or more other features
        in that entity."""
    rolling_function = False

    def __init__(self, *base_features):
        # Any edits made to this method should also be made to the
        # new_class_init method in make_trans_primitive
        self.base_features = [self._check_feature(f) for f in base_features]
        if any(bf.expanding for bf in self.base_features):
            self.expanding = True
        assert len(set([f.entity for f in self.base_features])) == 1, \
            "More than one entity for base features"
        super(TransformPrimitive, self).__init__(self.base_features[0].entity,
                                                 self.base_features)

    def generate_name(self):
        name = u"{}(".format(self.name.upper())
        name += u", ".join(f.get_name() for f in self.base_features)
        name += u")"
        return name

    @property
    def default_value(self):
        return self.base_features[0].default_value


def make_trans_primitive(function, input_types, return_type, name=None,
                         description='A custom transform primitive',
                         cls_attributes=None, uses_calc_time=False,
                         commutative=False):
    '''Returns a new transform primitive class

    Args:
        function (function): Function that takes in an array and applies some
            transformation to it, returning an array.

        input_types (list[Variable]): Variable types of the inputs.

        return_type (Variable): Variable type of return.

        name (str): Name of the primitive. If no name is provided, the name
            of `function` will be used.

        description (str): Description of primitive.

        cls_attributes (dict[str -> anytype]): Custom attributes to be added to
            class. Key is attribute name, value is the attribute value.

        uses_calc_time (bool): If True, the cutoff time the feature is being
            calculated at will be passed to the function as the keyword
            argument 'time'.

        commutative (bool): If True, will only make one feature per unique set
            of base features.

    Example:
        .. ipython :: python

            from featuretools.primitives import make_trans_primitive
            from featuretools.variable_types import Variable, Boolean

            def pd_is_in(array, list_of_outputs=None):
                if list_of_outputs is None:
                    list_of_outputs = []
                return pd.Series(array).isin(list_of_outputs)

            def isin_generate_name(self):
                return u"%s.isin(%s)" % (self.base_features[0].get_name(),
                                         str(self.kwargs['list_of_outputs']))

            IsIn = make_trans_primitive(
                function=pd_is_in,
                input_types=[Variable],
                return_type=Boolean,
                name="is_in",
                description="For each value of the base feature, checks "
                "whether it is in a list that provided.",
                cls_attributes={"generate_name": isin_generate_name})
    '''
    # dictionary that holds attributes for class
    cls = {"__doc__": description}
    if cls_attributes is not None:
        cls.update(cls_attributes)

    # creates the new class and set name and types
    name = name or function.__name__
    new_class = type(name, (TransformPrimitive,), cls)
    new_class.name = name
    new_class.input_types = input_types
    new_class.return_type = return_type
    new_class.commutative = commutative
    new_class, default_kwargs = inspect_function_args(new_class,
                                                      function,
                                                      uses_calc_time)

    if len(default_kwargs) > 0:
        new_class.default_kwargs = default_kwargs

        def new_class_init(self, *args, **kwargs):
            self.kwargs = copy.deepcopy(self.default_kwargs)
            self.base_features = [self._check_feature(f) for f in args]
            if any(bf.expanding for bf in self.base_features):
                self.expanding = True
            assert len(set([f.entity for f in self.base_features])) == 1, \
                "More than one entity for base features"
            self.kwargs.update(kwargs)
            self.partial = functools.partial(function, **self.kwargs)
            self.partial.__name__ = name

            super(TransformPrimitive, self).__init__(
                self.base_features[0].entity, self.base_features)
        new_class.__init__ = new_class_init
        new_class.get_function = lambda self: self.partial
    else:
        # creates a lambda function that returns function every time
        new_class.get_function = lambda self, f=function: f

    return new_class
