import pandas as pd

from ..base.primitive_base import PrimitiveBase

from featuretools.utils import is_string


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
        if left.dtype != object and is_string(right):
            left = left.astype(object)
    elif isinstance(right, pd.Series):
        if right.dtype != object and is_string(left):
            right = right.astype(object)
    return left, right
