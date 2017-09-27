from featuretools.variable_types import Numeric, Timedelta, Datetime, Variable, Boolean, TimeIndex
from .transform_primitive import TransformPrimitive
from .primitive_base import PrimitiveBase
import numpy as np
import operator
from .utils import apply_dual_op_from_feat


class BinaryFeature(TransformPrimitive):
    def __init__(self, left, right):
        if isinstance(left, (PrimitiveBase, Variable)):
            left = self._check_feature(left)

        if isinstance(right, (PrimitiveBase, Variable)):
            right = self._check_feature(right)

        base_features = []

        if isinstance(left, PrimitiveBase):
            base_features.append(left)
            self.left = left
        else:
            self.left = left

        self._right_index = -1
        if isinstance(right, PrimitiveBase):
            base_features.append(right)
            self.right = right
        else:
            self.right = right

        if len(base_features) == 0:
            raise ValueError("one of (left, right) must be an instance,"
                             "of PrimitiveBase or Variable")

        super(BinaryFeature, self).__init__(*base_features)

    def left_str(self):
        if isinstance(self.left, PrimitiveBase):
            return self.left.get_name()
        else:
            return self.left

    def right_str(self):
        if isinstance(self.right, PrimitiveBase):
            return self.right.get_name()
        else:
            return self.right

    @property
    def default_value(self):
        if isinstance(self.left, PrimitiveBase):
            left_default_val = self.left.default_value
        else:
            left_default_val = self.left
        if isinstance(self.right, PrimitiveBase):
            right_default_val = self.right.default_value
        else:
            right_default_val = self.right
        if np.isnan(left_default_val) or np.isnan(right_default_val):
            return np.nan
        else:
            return getattr(operator, self._get_op())(left_default_val,
                                                     right_default_val)

    def _get_name(self):
        return u"%s %s %s" % (self.left_str(),
                              self.operator, self.right_str())

    def _get_op(self):
        return self._operators[self.operator]

    def _get_rop(self):
        return self._roperators[self.operator]

    def hash(self):
        if isinstance(self.left, PrimitiveBase):
            left_hash = self.left.hash()
        else:
            left_hash = hash(self.left)
        if isinstance(self.right, PrimitiveBase):
            right_hash = self.right.hash()
        else:
            right_hash = hash(self.right)
        return hash(str(self.__class__) +
                    str(self.entity.id) +
                    str(self.operator) +
                    str(left_hash) +
                    str(right_hash))

    def pd_binary(self, array_1, array_2=None):
        return apply_dual_op_from_feat(self, array_1, array_2).values


class ArithmeticFeature(BinaryFeature):
    _ADD = '+'
    _SUB = '-'
    _MUL = '*'
    _DIV = '/'
    _MOD = '%'
    _operators = {
        _ADD: "__add__",
        _SUB: "__sub__",
        _MUL: "__mul__",
        _DIV: "__div__",
        _MOD: "__mod__",
    }

    _roperators = {
        _ADD: "__radd__",
        _SUB: "__rsub__",
        _MUL: "__rmul__",
        _DIV: "__rdiv__",
        _MOD: "__rmod__",
    }

    name = None
    input_types = [[Numeric, Numeric],
                   [Numeric]]
    return_type = Numeric
    operator = None

    @property
    def variable_type(self):
        dt_types = (Datetime, TimeIndex)
        td = Timedelta
        # TODO: separate this into a function
        if not isinstance(self.left, PrimitiveBase):
            if issubclass(self.right.variable_type, dt_types):
                return Timedelta
            elif self.right.variable_type == td:
                if self.left.dtype.name.find('datetime') > -1:
                    return Datetime
                else:
                    return Timedelta
            return self.right.variable_type
        if not isinstance(self.right, PrimitiveBase):
            if issubclass(self.left.variable_type, dt_types):
                return Timedelta
            elif self.left.variable_type == td:
                if self.right.dtype.name.find('datetime') > -1:
                    return Datetime
                else:
                    return Timedelta
            return self.left.variable_type
        left_vtype = self.left.variable_type
        right_vtype = self.right.variable_type
        if issubclass(left_vtype, dt_types) and issubclass(right_vtype, dt_types):
            return Timedelta
        elif issubclass(left_vtype, dt_types) or issubclass(right_vtype, dt_types):
            return Datetime
        elif left_vtype == td or right_vtype == td:
            return Timedelta
        else:
            return Numeric

    def get_function(self):
        return self.pd_binary


class Add(ArithmeticFeature):
    """Creates a transform feature that adds two features"""
    operator = ArithmeticFeature._ADD
    associative = True
    input_types = [[Numeric, Numeric],
                   [Numeric],
                   [TimeIndex],
                   [Datetime],
                   [Timedelta],
                   [TimeIndex, Timedelta],
                   [Timedelta, TimeIndex],
                   [Timedelta, Timedelta],
                   [Datetime, Timedelta],
                   [Timedelta, Datetime],
                   ]


class Subtract(ArithmeticFeature):
    """Creates a transform feature that subtracts two features"""
    operator = ArithmeticFeature._SUB
    input_types = [[Numeric, Numeric],
                   [Numeric],
                   [TimeIndex],
                   [Datetime],
                   [Timedelta],
                   [TimeIndex, Timedelta],
                   [Timedelta, TimeIndex],
                   [Timedelta, Timedelta],
                   [Datetime, Timedelta],
                   [Timedelta, Datetime],
                   ]


class Multiply(ArithmeticFeature):
    """Creates a transform feature that multplies two features"""
    operator = ArithmeticFeature._MUL
    associative = True


class Divide(ArithmeticFeature):
    """Creates a transform feature that divides two features"""
    operator = ArithmeticFeature._DIV


class Mod(ArithmeticFeature):
    """Creates a transform feature that divides two features"""
    operator = ArithmeticFeature._MOD


class Negate(Subtract):
    """Creates a transform feature that negates a feature"""
    input_types = [Numeric]

    def __init__(self, f):
        super(Negate, self).__init__(0, f)

    def _get_name(self):
        return u"-%s" % (self.right_str())


class Compare(BinaryFeature):
    """Compares two features using provided operator
        returns a boolean value"""
    EQ = '='
    NE = '!='
    LT = '<'
    GT = '>'
    LE = '<='
    GE = '>='

    _operators = {
        EQ: "__eq__",
        NE: "__ne__",
        LT: "__lt__",
        GT: "__gt__",
        LE: "__le__",
        GE: "__ge__"
    }

    _inv_operators = {
        EQ: NE,
        NE: EQ,
        LT: GE,
        GT: LE,
        LE: GT,
        GE: LT
    }

    _roperators = {
        EQ: "__eq__",
        NE: "__ne__",
        LT: "__ge__",
        GT: "__le__",
        LE: "__gt__",
        GE: "__lt__"
    }

    name = "compare"
    input_types = [[Variable], [Variable, Variable]]
    return_type = Boolean

    def __init__(self, left, operator, right):
        self.operator = operator
        if operator in ['=', '!=']:
            self.associative = True
        super(Compare, self).__init__(left, right)

    def invert(self):
        self.operator = self._inv_operators[self.operator]

    def get_function(self):
        return self.pd_binary


class And(TransformPrimitive):
    name = "and"
    input_types = [Boolean, Boolean]
    return_type = Boolean
    associative = True

    def get_function(self):
        return lambda left, right: np.logical_and(left, right)


class Or(TransformPrimitive):
    name = "or"
    input_types = [Boolean, Boolean]
    return_type = Boolean
    associative = True

    def get_function(self):
        return lambda left, right: np.logical_or(left, right)
