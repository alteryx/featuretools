from datetime import datetime

import pandas as pd
from pytest import raises

from featuretools.primitives import IsNull, Max
from featuretools.primitives.base import PrimitiveBase, make_agg_primitive
from featuretools.variable_types import DatetimeTimeIndex, Numeric


def test_call_agg():
    primitive = Max()

    # the assert is run twice on purpose
    assert 5 == primitive(range(6))
    assert 5 == primitive(range(6))


def test_call_trans():
    primitive = IsNull()
    assert pd.Series([False for i in range(6)]).equals(primitive(range(6)))
    assert pd.Series([False for i in range(6)]).equals(primitive(range(6)))


def test_uses_calc_time():
    def time_since_last(values, time=None):
        time_since = time - values.iloc[0]
        return time_since.total_seconds()

    TimeSinceLast = make_agg_primitive(time_since_last,
                                       [DatetimeTimeIndex],
                                       Numeric,
                                       name="time_since_last",
                                       uses_calc_time=True)
    primitive = TimeSinceLast()
    datetimes = pd.Series([datetime(2015, 6, 7), datetime(2015, 6, 6)])
    answer = 86400.0
    assert answer == primitive(datetimes, time=datetime(2015, 6, 8))


def test_call_multiple_args():
    class TestPrimitive(PrimitiveBase):
        def get_function(self):
            def test(x, y):
                return y
            return test
    primitive = TestPrimitive()
    assert pd.Series([0, 1]).equals(primitive(range(1), range(2)))
    assert pd.Series([0, 1]).equals(primitive(range(1), range(2)))


def test_get_function_called_once():
    class TestPrimitive(PrimitiveBase):
        def __init__(self):
            self.get_function_call_count = 0

        def get_function(self):
            self.get_function_call_count += 1

            def test(x):
                return x
            return test

    primitive = TestPrimitive()
    primitive(range(6))
    primitive(range(6))
    assert primitive.get_function_call_count == 1


def test_args_string():
    class Primitive(PrimitiveBase):
        def __init__(self, bool=True, int=0, float=None):
            self.bool = bool
            self.int = int
            self.float = float

    primitive = Primitive(bool=True, int=4, float=.1)
    string = primitive.get_args_string()
    assert string == ', int=4, float=0.1'


def test_args_string_default():
    class Primitive(PrimitiveBase):
        def __init__(self, bool=True, int=0, float=None):
            self.bool = bool
            self.int = int
            self.float = float

    string = Primitive().get_args_string()
    assert string == ''


def test_args_string_mixed():
    class Primitive(PrimitiveBase):
        def __init__(self, bool=True, int=0, float=None):
            self.bool = bool
            self.int = int
            self.float = float

    primitive = Primitive(bool=False, int=0)
    string = primitive.get_args_string()
    assert string == ', bool=False'


def test_args_string_undefined():
    class Primitive(PrimitiveBase):
        pass

    string = Primitive().get_args_string()
    assert string == ''


def test_args_string_error():
    class Primitive(PrimitiveBase):
        def __init__(self, bool=True, int=0, float=None):
            pass

    with raises(AssertionError, match='must be attribute'):
        Primitive(bool=True, int=4, float=.1).get_args_string()
