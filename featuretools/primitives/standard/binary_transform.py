from builtins import str

import numpy as np
import pandas as pd

from ..base.transform_primitive_base import TransformPrimitive

from featuretools.variable_types import (
    Boolean,
    Datetime,
    Numeric,
    Ordinal,
    Variable
)


class GreaterThan(TransformPrimitive):
    """Determines if values in one list are greater than another list.

    Description:
        Given a list of values X and a list of values Y, determine
        whether each value in X is greater than each corresponding
        value in Y. Equal pairs will return `False`.

    Examples:
        >>> greater_than = GreaterThan()
        >>> greater_than([2, 1, 2], [1, 2, 2]).tolist()
        [True, False, False]
    """
    name = "greater_than"
    input_types = [[Numeric, Numeric], [Datetime, Datetime], [Ordinal, Ordinal]]
    return_type = Boolean

    def get_function(self):
        return np.greater

    def generate_name(self, base_feature_names):
        return "%s > %s" % (base_feature_names[0], base_feature_names[1])


class GreaterThanScalar(TransformPrimitive):
    """Determines if values are greater than a given scalar.

    Description:
        Given a list of values and a constant scalar, determine
        whether each of the values is greater than the scalar.
        If a value is equal to the scalar, return `False`.

    Examples:
        >>> greater_than_scalar = GreaterThanScalar(value=2)
        >>> greater_than_scalar([3, 1, 2]).tolist()
        [True, False, False]
    """
    name = "greater_than_scalar"
    input_types = [[Numeric], [Datetime], [Ordinal]]
    return_type = Boolean

    def __init__(self, value=0):
        self.value = value

    def get_function(self):
        def greater_than_scalar(vals):
            # convert series to handle both numeric and datetime case
            return pd.Series(vals) > self.value
        return greater_than_scalar

    def generate_name(self, base_feature_names):
        return "%s > %s" % (base_feature_names[0], str(self.value))


class GreaterThanEqualTo(TransformPrimitive):
    """Determines if values in one list are greater than or equal to another list.

    Description:
        Given a list of values X and a list of values Y, determine
        whether each value in X is greater than or equal to each
        corresponding value in Y. Equal pairs will return `True`.

    Examples:
        >>> greater_than_equal_to = GreaterThanEqualTo()
        >>> greater_than_equal_to([2, 1, 2], [1, 2, 2]).tolist()
        [True, False, True]
    """
    name = "greater_than_equal_to"
    input_types = [[Numeric, Numeric], [Datetime, Datetime], [Ordinal, Ordinal]]
    return_type = Boolean

    def get_function(self):
        return np.greater_equal

    def generate_name(self, base_feature_names):
        return "%s >= %s" % (base_feature_names[0], base_feature_names[1])


class GreaterThanEqualToScalar(TransformPrimitive):
    """Determines if values are greater than or equal to a given scalar.

    Description:
        Given a list of values and a constant scalar, determine
        whether each of the values is greater than or equal to the
        scalar. If a value is equal to the scalar, return `True`.

    Examples:
        >>> greater_than_equal_to_scalar = GreaterThanEqualToScalar(value=2)
        >>> greater_than_equal_to_scalar([3, 1, 2]).tolist()
        [True, False, True]
    """
    name = "greater_than_equal_to_scalar"
    input_types = [[Numeric], [Datetime], [Ordinal]]
    return_type = Boolean

    def __init__(self, value=0):
        self.value = value

    def get_function(self):
        def greater_than_equal_to_scalar(vals):
            # convert series to handle both numeric and datetime case
            return pd.Series(vals) >= self.value
        return greater_than_equal_to_scalar

    def generate_name(self, base_feature_names):
        return "%s >= %s" % (base_feature_names[0], str(self.value))


class LessThan(TransformPrimitive):
    """Determines if values in one list are less than another list.

    Description:
        Given a list of values X and a list of values Y, determine
        whether each value in X is less than each corresponding value
        in Y. Equal pairs will return `False`.

    Examples:
        >>> less_than = LessThan()
        >>> less_than([2, 1, 2], [1, 2, 2]).tolist()
        [False, True, False]
    """
    name = "less_than"
    input_types = [[Numeric, Numeric], [Datetime, Datetime], [Ordinal, Ordinal]]
    return_type = Boolean

    def get_function(self):
        return np.less

    def generate_name(self, base_feature_names):
        return "%s < %s" % (base_feature_names[0], base_feature_names[1])


class LessThanScalar(TransformPrimitive):
    """Determines if values are less than a given scalar.

    Description:
        Given a list of values and a constant scalar, determine
        whether each of the values is less than the scalar.
        If a value is equal to the scalar, return `False`.

    Examples:
        >>> less_than_scalar = LessThanScalar(value=2)
        >>> less_than_scalar([3, 1, 2]).tolist()
        [False, True, False]
    """
    name = "less_than_scalar"
    input_types = [[Numeric], [Datetime], [Ordinal]]
    return_type = Boolean

    def __init__(self, value=0):
        self.value = value

    def get_function(self):
        def less_than_scalar(vals):
            # convert series to handle both numeric and datetime case
            return pd.Series(vals) < self.value
        return less_than_scalar

    def generate_name(self, base_feature_names):
        return "%s < %s" % (base_feature_names[0], str(self.value))


class LessThanEqualTo(TransformPrimitive):
    """Determines if values in one list are less than or equal to another list.

    Description:
        Given a list of values X and a list of values Y, determine
        whether each value in X is less than or equal to each
        corresponding value in Y. Equal pairs will return `True`.

    Examples:
        >>> less_than_equal_to = LessThanEqualTo()
        >>> less_than_equal_to([2, 1, 2], [1, 2, 2]).tolist()
        [False, True, True]
    """
    name = "less_than_equal_to"
    input_types = [[Numeric, Numeric], [Datetime, Datetime], [Ordinal, Ordinal]]
    return_type = Boolean

    def get_function(self):
        return np.less_equal

    def generate_name(self, base_feature_names):
        return "%s <= %s" % (base_feature_names[0], base_feature_names[1])


class LessThanEqualToScalar(TransformPrimitive):
    """Determines if values are less than or equal to a given scalar.

    Description:
        Given a list of values and a constant scalar, determine
        whether each of the values is less than or equal to the
        scalar. If a value is equal to the scalar, return `True`.

    Examples:
        >>> less_than_equal_to_scalar = LessThanEqualToScalar(value=2)
        >>> less_than_equal_to_scalar([3, 1, 2]).tolist()
        [False, True, True]
    """
    name = "less_than_equal_to_scalar"
    input_types = [[Numeric], [Datetime], [Ordinal]]
    return_type = Boolean

    def __init__(self, value=0):
        self.value = value

    def get_function(self):
        def less_than_equal_to_scalar(vals):
            # convert series to handle both numeric and datetime case
            return pd.Series(vals) <= self.value
        return less_than_equal_to_scalar

    def generate_name(self, base_feature_names):
        return "%s <= %s" % (base_feature_names[0], str(self.value))


class Equal(TransformPrimitive):
    """Determines if values in one list are equal to another list.

    Description:
        Given a list of values X and a list of values Y, determine
        whether each value in X is equal to each corresponding value
        in Y.

    Examples:
        >>> equal = Equal()
        >>> equal([2, 1, 2], [1, 2, 2]).tolist()
        [False, False, True]
    """
    name = "equal"
    input_types = [Variable, Variable]
    return_type = Boolean
    commutative = True

    def get_function(self):
        return np.equal

    def generate_name(self, base_feature_names):
        return "%s = %s" % (base_feature_names[0], base_feature_names[1])


class EqualScalar(TransformPrimitive):
    """Determines if values in a list are equal to a given scalar.

    Description:
        Given a list of values and a constant scalar, determine
        whether each of the values is equal to the scalar.

    Examples:
        >>> equal_scalar = EqualScalar(value=2)
        >>> equal_scalar([3, 1, 2]).tolist()
        [False, False, True]
    """
    name = "equal_scalar"
    input_types = [Variable]
    return_type = Boolean

    def __init__(self, value=None):
        self.value = value

    def get_function(self):
        def equal_scalar(vals):
            # case to correct pandas type for comparison
            return pd.Series(vals).astype(pd.Series([self.value]).dtype) == self.value
        return equal_scalar

    def generate_name(self, base_feature_names):
        return "%s = %s" % (base_feature_names[0], str(self.value))


class NotEqual(TransformPrimitive):
    """Determines if values in one list are not equal to another list.

    Description:
        Given a list of values X and a list of values Y, determine
        whether each value in X is not equal to each corresponding
        value in Y.

    Examples:
        >>> not_equal = NotEqual()
        >>> not_equal([2, 1, 2], [1, 2, 2]).tolist()
        [True, True, False]
    """
    name = "not_equal"
    input_types = [Variable, Variable]
    return_type = Boolean
    commutative = True

    def get_function(self):
        return np.not_equal

    def generate_name(self, base_feature_names):
        return "%s != %s" % (base_feature_names[0], base_feature_names[1])


class NotEqualScalar(TransformPrimitive):
    """Determines if values in a list are not equal to a given scalar.

    Description:
        Given a list of values and a constant scalar, determine
        whether each of the values is not equal to the scalar.

    Examples:
        >>> not_equal_scalar = NotEqualScalar(value=2)
        >>> not_equal_scalar([3, 1, 2]).tolist()
        [True, True, False]
    """
    name = "not_equal_scalar"
    input_types = [Variable]
    return_type = Boolean

    def __init__(self, value=None):
        self.value = value

    def get_function(self):
        def not_equal_scalar(vals):
            # case to correct pandas type for comparison
            return pd.Series(vals).astype(pd.Series([self.value]).dtype) != self.value
        return not_equal_scalar

    def generate_name(self, base_feature_names):
        return "%s != %s" % (base_feature_names[0], str(self.value))


class AddNumeric(TransformPrimitive):
    """Element-wise addition of two lists.

    Description:
        Given a list of values X and a list of values
        Y, determine the sum of each value in X with its
        corresponding value in Y.

    Examples:
        >>> add_numeric = AddNumeric()
        >>> add_numeric([2, 1, 2], [1, 2, 2]).tolist()
        [3, 3, 4]
    """
    name = "add_numeric"
    input_types = [Numeric, Numeric]
    return_type = Numeric
    commutative = True

    def get_function(self):
        return np.add

    def generate_name(self, base_feature_names):
        return "%s + %s" % (base_feature_names[0], base_feature_names[1])


class AddNumericScalar(TransformPrimitive):
    """Add a scalar to each value in the list.

    Description:
        Given a list of numeric values and a scalar, add
        the given scalar to each value in the list.

    Examples:
        >>> add_numeric_scalar = AddNumericScalar(value=2)
        >>> add_numeric_scalar([3, 1, 2]).tolist()
        [5, 3, 4]
    """
    name = "add_numeric_scalar"
    input_types = [Numeric]
    return_type = Numeric

    def __init__(self, value=0):
        self.value = value

    def get_function(self):
        def add_scalar(vals):
            return vals + self.value
        return add_scalar

    def generate_name(self, base_feature_names):
        return "%s + %s" % (base_feature_names[0], str(self.value))


class SubtractNumeric(TransformPrimitive):
    """Element-wise subtraction of two lists.

    Description:
        Given a list of values X and a list of values
        Y, determine the difference of each value
        in X from its corresponding value in Y.

    Args:
        commutative (bool): determines if Deep Feature Synthesis should
            generate both x - y and y - x, or just one. If True, there is no
            guarantee which of the two will be generated. Defaults to True.

    Examples:
        >>> subtract_numeric = SubtractNumeric()
        >>> subtract_numeric([2, 1, 2], [1, 2, 2]).tolist()
        [1, -1, 0]
    """
    name = "subtract_numeric"
    input_types = [Numeric, Numeric]
    return_type = Numeric

    def __init__(self, commutative=True):
        self.commutative = commutative

    def get_function(self):
        return np.subtract

    def generate_name(self, base_feature_names):
        return "%s - %s" % (base_feature_names[0], base_feature_names[1])


class SubtractNumericScalar(TransformPrimitive):
    """Subtract a scalar from each element in the list.

    Description:
        Given a list of numeric values and a scalar, subtract
        the given scalar from each value in the list.

    Examples:
        >>> subtract_numeric_scalar = SubtractNumericScalar(value=2)
        >>> subtract_numeric_scalar([3, 1, 2]).tolist()
        [1, -1, 0]
    """
    name = "subtract_numeric_scalar"
    input_types = [Numeric]
    return_type = Numeric

    def __init__(self, value=0):
        self.value = value

    def get_function(self):
        def subtract_scalar(vals):
            return vals - self.value
        return subtract_scalar

    def generate_name(self, base_feature_names):
        return "%s - %s" % (base_feature_names[0], str(self.value))


class ScalarSubtractNumericFeature(TransformPrimitive):
    """Subtract each value in the list from a given scalar.

    Description:
        Given a list of numeric values and a scalar, subtract
        the each value from the scalar and return the list of
        differences.

    Examples:
        >>> scalar_subtract_numeric_feature = ScalarSubtractNumericFeature(value=2)
        >>> scalar_subtract_numeric_feature([3, 1, 2]).tolist()
        [-1, 1, 0]
    """
    name = "scalar_subtract_numeric_feature"
    input_types = [Numeric]
    return_type = Numeric

    def __init__(self, value=0):
        self.value = value

    def get_function(self):
        def scalar_subtract_numeric_feature(vals):
            return self.value - vals
        return scalar_subtract_numeric_feature

    def generate_name(self, base_feature_names):
        return "%s - %s" % (str(self.value), base_feature_names[0])


class MultiplyNumeric(TransformPrimitive):
    """Element-wise multiplication of two lists.

    Description:
        Given a list of values X and a list of values
        Y, determine the product of each value in X
        with its corresponding value in Y.

    Examples:
        >>> multiply_numeric = MultiplyNumeric()
        >>> multiply_numeric([2, 1, 2], [1, 2, 2]).tolist()
        [2, 2, 4]
    """
    name = "multiply_numeric"
    input_types = [Numeric, Numeric]
    return_type = Numeric
    commutative = True

    def get_function(self):
        return np.multiply

    def generate_name(self, base_feature_names):
        return "%s * %s" % (base_feature_names[0], base_feature_names[1])


class MultiplyNumericScalar(TransformPrimitive):
    """Multiply each element in the list by a scalar.

    Description:
        Given a list of numeric values and a scalar, multiply
        each value in the list by the scalar.

    Examples:
        >>> multiply_numeric_scalar = MultiplyNumericScalar(value=2)
        >>> multiply_numeric_scalar([3, 1, 2]).tolist()
        [6, 2, 4]
    """
    name = "multiply_numeric_scalar"
    input_types = [Numeric]
    return_type = Numeric

    def __init__(self, value=1):
        self.value = value

    def get_function(self):
        def multiply_scalar(vals):
            return vals * self.value
        return multiply_scalar

    def generate_name(self, base_feature_names):
        return "%s * %s" % (base_feature_names[0], str(self.value))


class DivideNumeric(TransformPrimitive):
    """Element-wise division of two lists.

    Description:
        Given a list of values X and a list of values
        Y, determine the quotient of each value in X
        divided by its corresponding value in Y.

    Args:
        commutative (bool): determines if Deep Feature Synthesis should
            generate both x / y and y / x, or just one. If True, there is
            no guarantee which of the two will be generated. Defaults to False.

    Examples:
        >>> divide_numeric = DivideNumeric()
        >>> divide_numeric([2.0, 1.0, 2.0], [1.0, 2.0, 2.0]).tolist()
        [2.0, 0.5, 1.0]
    """
    name = "divide_numeric"
    input_types = [Numeric, Numeric]
    return_type = Numeric

    def __init__(self, commutative=False):
        self.commutative = commutative

    def get_function(self):
        return np.divide

    def generate_name(self, base_feature_names):
        return "%s / %s" % (base_feature_names[0], base_feature_names[1])


class DivideNumericScalar(TransformPrimitive):
    """Divide each element in the list by a scalar.

    Description:
        Given a list of numeric values and a scalar, divide
        each value in the list by the scalar.

    Examples:
        >>> divide_numeric_scalar = DivideNumericScalar(value=2)
        >>> divide_numeric_scalar([3, 1, 2]).tolist()
        [1.5, 0.5, 1.0]
    """
    name = "divide_numeric_scalar"
    input_types = [Numeric]
    return_type = Numeric

    def __init__(self, value=1):
        self.value = value

    def get_function(self):
        def divide_scalar(vals):
            return vals / self.value
        return divide_scalar

    def generate_name(self, base_feature_names):
        return "%s / %s" % (base_feature_names[0], str(self.value))


class DivideByFeature(TransformPrimitive):
    """Divide a scalar by each value in the list.

    Description:
        Given a list of numeric values and a scalar, divide
        the scalar by each value and return the list of
        quotients.

    Examples:
        >>> divide_by_feature = DivideByFeature(value=2)
        >>> divide_by_feature([4, 1, 2]).tolist()
        [0.5, 2.0, 1.0]
    """
    name = "divide_by_feature"
    input_types = [Numeric]
    return_type = Numeric

    def __init__(self, value=1):
        self.value = value

    def get_function(self):
        def divide_by_feature(vals):
            return self.value / vals
        return divide_by_feature

    def generate_name(self, base_feature_names):
        return "%s / %s" % (str(self.value), base_feature_names[0])


class ModuloNumeric(TransformPrimitive):
    """Element-wise modulo of two lists.

    Description:
        Given a list of values X and a list of values Y,
        determine the modulo, or remainder of each value in
        X after it's divided by its corresponding value in Y.

    Examples:
        >>> modulo_numeric = ModuloNumeric()
        >>> modulo_numeric([2, 1, 5], [1, 2, 2]).tolist()
        [0, 1, 1]
    """
    name = "modulo_numeric"
    input_types = [Numeric, Numeric]
    return_type = Numeric

    def get_function(self):
        return np.mod

    def generate_name(self, base_feature_names):
        return "%s %% %s" % (base_feature_names[0], base_feature_names[1])


class ModuloNumericScalar(TransformPrimitive):
    """Return the modulo of each element in the list by a scalar.

    Description:
        Given a list of numeric values and a scalar, return
        the modulo, or remainder of each value after being
        divided by the scalar.

    Examples:
        >>> modulo_numeric_scalar = ModuloNumericScalar(value=2)
        >>> modulo_numeric_scalar([3, 1, 2]).tolist()
        [1, 1, 0]
    """
    name = "modulo_numeric_scalar"
    input_types = [Numeric]
    return_type = Numeric

    def __init__(self, value=1):
        self.value = value

    def get_function(self):
        def modulo_scalar(vals):
            return vals % self.value
        return modulo_scalar

    def generate_name(self, base_feature_names):
        return "%s %% %s" % (base_feature_names[0], str(self.value))


class ModuloByFeature(TransformPrimitive):
    """Return the modulo of a scalar by each element in the list.

    Description:
        Given a list of numeric values and a scalar, return the
        modulo, or remainder of the scalar after being divided
        by each value.

    Examples:
        >>> modulo_by_feature = ModuloByFeature(value=2)
        >>> modulo_by_feature([4, 1, 2]).tolist()
        [2, 0, 0]
    """
    name = "modulo_by_feature"
    input_types = [Numeric]
    return_type = Numeric

    def __init__(self, value=1):
        self.value = value

    def get_function(self):
        def modulo_by_feature(vals):
            return self.value % vals
        return modulo_by_feature

    def generate_name(self, base_feature_names):
        return "%s %% %s" % (str(self.value), base_feature_names[0])


class And(TransformPrimitive):
    """Element-wise logical AND of two lists.

    Description:
        Given a list of booleans X and a list of booleans Y,
        determine whether each value in X is `True`, and
        whether its corresponding value in Y is also `True`.

    Examples:
        >>> _and = And()
        >>> _and([False, True, False], [True, True, False]).tolist()
        [False, True, False]
    """
    name = "and"
    input_types = [Boolean, Boolean]
    return_type = Boolean
    commutative = True

    def get_function(self):
        return np.logical_and

    def generate_name(self, base_feature_names):
        return "AND(%s, %s)" % (base_feature_names[0], base_feature_names[1])


class Or(TransformPrimitive):
    """Element-wise logical OR of two lists.

    Description:
        Given a list of booleans X and a list of booleans Y,
        determine whether each value in X is `True`, or
        whether its corresponding value in Y is `True`.

    Examples:
        >>> _or = Or()
        >>> _or([False, True, False], [True, True, False]).tolist()
        [True, True, False]
    """
    name = "or"
    input_types = [Boolean, Boolean]
    return_type = Boolean
    commutative = True

    def get_function(self):
        return np.logical_or

    def generate_name(self, base_feature_names):
        return "OR(%s, %s)" % (base_feature_names[0], base_feature_names[1])
