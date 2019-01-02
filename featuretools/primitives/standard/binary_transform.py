from builtins import str

import numpy as np

from ..base.transform_primitive_base import TransformPrimitive
from featuretools.variable_types import Boolean, Numeric, Variable, Datetime


class GreaterThan(TransformPrimitive):
    name = "greater_than"
    input_types = [[Numeric, Numeric], [Datetime, Datetime]]
    return_type = Boolean

    def get_function(self):
        return np.greater

    def generate_name(self, base_feature_names):
        return "%s > %s" % (base_feature_names[0], base_feature_names[1])


class GreaterThanScalar(TransformPrimitive):
    name = "greater_than_scalar"
    input_types = [[Numeric], [Datetime]]
    return_type = Boolean

    def __init__(self, value=None):
        self.value = value

    def get_function(self):
        def greater_than_scalar(vals):
            return vals > self.value
        return greater_than_scalar

    def generate_name(self, base_feature_names):
        return "%s > %s" % (base_feature_names[0], str(self.value))


class GreaterThanEqualTo(TransformPrimitive):
    name = "greater_than_equal_to"
    input_types = [[Numeric, Numeric], [Datetime, Datetime]]
    return_type = Boolean

    def get_function(self):
        return np.greater_equal

    def generate_name(self, base_feature_names):
        return "%s >= %s" % (base_feature_names[0], base_feature_names[1])


class GreaterThanEqualToScalar(TransformPrimitive):
    name = "greater_than_equal_to_scalar"
    input_types = [[Numeric], [Datetime]]
    return_type = Boolean

    def __init__(self, value=None):
        self.value = value

    def get_function(self):
        def greater_than_equal_to_scalar(vals):
            return vals >= self.value
        return greater_than_equal_to_scalar

    def generate_name(self, base_feature_names):
        return "%s >= %s" % (base_feature_names[0], str(self.value))


class LessThan(TransformPrimitive):
    name = "less_than"
    input_types = [[Numeric, Numeric], [Datetime, Datetime]]
    return_type = Boolean

    def get_function(self):
        return np.less

    def generate_name(self, base_feature_names):
        return "%s < %s" % (base_feature_names[0], base_feature_names[1])


class LessThanScalar(TransformPrimitive):
    name = "less_than_scalar"
    input_types = [[Numeric], [Datetime]]
    return_type = Boolean

    def __init__(self, value=None):
        self.value = value

    def get_function(self):
        def less_than_scalar(vals):
            return vals < self.value
        return less_than_scalar

    def generate_name(self, base_feature_names):
        return "%s < %s" % (base_feature_names[0], str(self.value))


class LessThanEqualTo(TransformPrimitive):
    name = "less_than_equal_to"
    input_types = [[Numeric, Numeric], [Datetime, Datetime]]
    return_type = Boolean

    def get_function(self):
        return np.less_equal

    def generate_name(self, base_feature_names):
        return "%s <= %s" % (base_feature_names[0], base_feature_names[1])


class LessThanEqualToScalar(TransformPrimitive):
    name = "less_than_equal_to_scalar"
    input_types = [[Numeric], [Datetime]]
    return_type = Boolean

    def __init__(self, value=None):
        self.value = value

    def get_function(self):
        def less_than_equal_to_scalar(vals):
            return vals <= self.value
        return less_than_equal_to_scalar

    def generate_name(self, base_feature_names):
        return "%s <= %s" % (base_feature_names[0], str(self.value))


class Equal(TransformPrimitive):
    name = "equal"
    input_types = [Variable, Variable]
    return_type = Boolean
    commutative = True

    def get_function(self):
        return np.equal

    def generate_name(self, base_feature_names):
        return "%s = %s" % (base_feature_names[0], base_feature_names[1])


class EqualScalar(TransformPrimitive):
    name = "equal_scalar"
    input_types = [Variable]
    return_type = Boolean
    commutative = True

    def __init__(self, value=None):
        self.value = value

    def get_function(self):
        def equal_scalar(vals):
            return vals == self.value
        return equal_scalar

    def generate_name(self, base_feature_names):
        return "%s = %s" % (base_feature_names[0], str(self.value))


class NotEqual(TransformPrimitive):
    name = "not_equal"
    input_types = [Variable, Variable]
    return_type = Boolean
    commutative = True

    def get_function(self):
        return np.not_equal

    def generate_name(self, base_feature_names):
        return "%s != %s" % (base_feature_names[0], base_feature_names[1])


class NotEqualScalar(TransformPrimitive):
    name = "not_equal_scalar"
    input_types = [Variable]
    return_type = Boolean
    commutative = True

    def __init__(self, value=None):
        self.value = value

    def get_function(self):
        def not_equal_scalar(vals):
            return vals != self.value
        return not_equal_scalar

    def generate_name(self, base_feature_names):
        return "%s != %s" % (base_feature_names[0], str(self.value))


class AddNumeric(TransformPrimitive):
    name = "add_numeric"
    input_types = [Numeric, Numeric]
    return_type = Numeric
    commutative = True

    def get_function(self):
        return np.add

    def generate_name(self, base_feature_names):
        return "%s + %s" % (base_feature_names[0], base_feature_names[1])


class AddNumericScalar(TransformPrimitive):
    name = "add_numeric_scalar"
    input_types = [Numeric]
    return_type = Numeric
    commutative = True

    def __init__(self, value=None):
        self.value = value

    def get_function(self):
        def add_scalar(vals):
            return vals + self.value
        return add_scalar

    def generate_name(self, base_feature_names):
        return "%s + %s" % (base_feature_names[0], str(self.value))


class SubtractNumeric(TransformPrimitive):
    name = "subtract_numeric"
    input_types = [Numeric, Numeric]
    return_type = Numeric
    commutative = True

    def get_function(self):
        return np.subtract

    def generate_name(self, base_feature_names):
        return "%s - %s" % (base_feature_names[0], base_feature_names[1])


class SubtractNumericScalar(TransformPrimitive):
    name = "subtract_numeric_scalar"
    input_types = [Numeric]
    return_type = Numeric
    commutative = True

    def __init__(self, value=None):
        self.value = value

    def get_function(self):
        def subtract_scalar(vals):
            return vals - self.value
        return subtract_scalar

    def generate_name(self, base_feature_names):
        return "%s - %s" % (base_feature_names[0], str(self.value))


class MultiplyNumeric(TransformPrimitive):
    name = "multiply_numeric"
    input_types = [Numeric, Numeric]
    return_type = Numeric
    commutative = True

    def get_function(self):
        return np.multiply

    def generate_name(self, base_feature_names):
        return "%s * %s" % (base_feature_names[0], base_feature_names[1])


class MultiplyNumericScalar(TransformPrimitive):
    name = "multiply_numeric_scalar"
    input_types = [Numeric]
    return_type = Numeric
    commutative = True

    def __init__(self, value=None):
        self.value = value

    def get_function(self):
        def multiply_scalar(vals):
            return vals * self.value
        return multiply_scalar

    def generate_name(self, base_feature_names):
        return "%s * %s" % (base_feature_names[0], str(self.value))


class DivideNumeric(TransformPrimitive):
    name = "divide_numeric"
    input_types = [Numeric, Numeric]
    return_type = Numeric

    def get_function(self):
        return np.divide

    def generate_name(self, base_feature_names):
        return "%s / %s" % (base_feature_names[0], base_feature_names[1])


class DivideNumericScalar(TransformPrimitive):
    name = "divide_numeric_scalar"
    input_types = [Numeric]
    return_type = Numeric

    def __init__(self, value=None):
        self.value = value

    def get_function(self):
        def divide_scalar(vals):
            return vals / self.value
        return divide_scalar

    def generate_name(self, base_feature_names):
        return "%s / %s" % (base_feature_names[0], str(self.value))


class ModuloNumeric(TransformPrimitive):
    name = "modulo_numeric"
    input_types = [Numeric, Numeric]
    return_type = Numeric

    def get_function(self):
        return np.mod

    def generate_name(self, base_feature_names):
        return "%s %% %s" % (base_feature_names[0], base_feature_names[1])


class ModuloNumericScalar(TransformPrimitive):
    name = "modulo_numeric"
    input_types = [Numeric]
    return_type = Numeric

    def __init__(self, value=None):
        self.value = value

    def get_function(self):
        def modulo_scalar(vals):
            return vals % self.value
        return modulo_scalar

    def generate_name(self, base_feature_names):
        return "%s %% %s" % (base_feature_names[0], str(self.value))


class And(TransformPrimitive):
    name = "and"
    input_types = [Boolean, Boolean]
    return_type = Boolean
    commutative = True

    def get_function(self):
        return np.logical_and

    def generate_name(self, base_feature_names):
        return "AND(%s, %s)" % (base_feature_names[0], base_feature_names[1])


class Or(TransformPrimitive):
    name = "or"
    input_types = [Boolean, Boolean]
    return_type = Boolean
    commutative = True

    def get_function(self):
        return np.logical_or

    def generate_name(self, base_feature_names):
        return "OR(%s, %s)" % (base_feature_names[0], base_feature_names[1])
