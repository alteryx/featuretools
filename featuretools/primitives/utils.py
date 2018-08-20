from inspect import getargspec, isclass

import pandas as pd
from past.builtins import basestring

from .primitive_base import PrimitiveBase

import featuretools.primitives


def apply_dual_op_from_feat(f, array_1, array_2=None):
    left = f.left
    right = f.right
    left_array = array_1
    if array_2 is not None:
        right_array = array_2
    else:
        right_array = array_1
    to_op = None
    other = None
    if isinstance(left, PrimitiveBase):
        left = pd.Series(left_array)
        other = right
        to_op = left
        op = f._get_op()
    if isinstance(right, PrimitiveBase):
        right = pd.Series(right_array)
        other = right
        if to_op is None:
            other = left
            to_op = right
            op = f._get_rop()
    to_op, other = ensure_compatible_dtype(to_op, other)
    op = getattr(to_op, op)

    assert op is not None, \
        "Need at least one feature for dual op, found 2 scalars"
    return op(other)


def get_aggregation_primitives():
    aggregation_primitives = set([])
    for attribute_string in dir(featuretools.primitives):
        attribute = getattr(featuretools.primitives, attribute_string)
        if isclass(attribute):
            if issubclass(attribute,
                          featuretools.primitives.AggregationPrimitive):
                if attribute.name:
                    aggregation_primitives.add(attribute)
    return {prim.name.lower(): prim for prim in aggregation_primitives}


def get_transform_primitives():
    transform_primitives = set([])
    for attribute_string in dir(featuretools.primitives):
        attribute = getattr(featuretools.primitives, attribute_string)
        if isclass(attribute):
            if issubclass(attribute,
                          featuretools.primitives.TransformPrimitive):
                if attribute.name:
                    transform_primitives.add(attribute)
    return {prim.name.lower(): prim for prim in transform_primitives}


def list_primitives():
    transform_primitives = get_transform_primitives()
    agg_primitives = get_aggregation_primitives()
    transform_df = pd.DataFrame({'name': list(transform_primitives.keys()),
                                 'description': [prim.__doc__.split("\n")[0] for prim in transform_primitives.values()]})
    transform_df['type'] = 'transform'
    agg_df = pd.DataFrame({'name': list(agg_primitives.keys()),
                           'description': [prim.__doc__.split("\n")[0] for prim in agg_primitives.values()]})
    agg_df['type'] = 'aggregation'

    return pd.concat([agg_df, transform_df], ignore_index=True)[['name', 'type', 'description']]


def ensure_compatible_dtype(left, right):
    # Pandas converts dtype to float
    # if all nans. If the actual values are
    # strings/objects though, future features
    # that depend on these values may error
    # unless we explicitly set the dtype to object
    if isinstance(left, pd.Series) and isinstance(right, pd.Series):
        if left.dtype != object and right.dtype == object:
            left = left.astype(object)
        elif right.dtype != object and left.dtype == object:
            right = right.astype(object)
    elif isinstance(left, pd.Series):
        if left.dtype != object and isinstance(right, basestring):
            left = left.astype(object)
    elif isinstance(right, pd.Series):
        if right.dtype != object and isinstance(left, basestring):
            right = right.astype(object)
    return left, right


def inspect_function_args(new_class, function, uses_calc_time):
    # inspect function to see if there are keyword arguments
    argspec = getargspec(function)
    kwargs = {}
    if argspec.defaults is not None:
        lowest_kwargs_position = len(argspec.args) - len(argspec.defaults)

    for i, arg in enumerate(argspec.args):
        if arg == 'time':
            if not uses_calc_time:
                raise ValueError("'time' is a restricted keyword.  Please"
                                 " use a different keyword.")
            else:
                new_class.uses_calc_time = True
        if argspec.defaults is not None and i >= lowest_kwargs_position:
            kwargs[arg] = argspec.defaults[i - lowest_kwargs_position]
    return new_class, kwargs
