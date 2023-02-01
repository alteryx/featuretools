from datetime import datetime

import numpy as np
import pandas as pd
from pytest import raises

from featuretools.primitives import Haversine, IsIn, IsNull, Max, TimeSinceLast
from featuretools.primitives.base import TransformPrimitive


def test_call_agg():
    primitive = Max()

    # the assert is run twice on purpose
    for _ in range(2):
        assert 5 == primitive(range(6))


def test_call_trans():
    primitive = IsNull()
    for _ in range(2):
        assert pd.Series([False] * 6).equals(primitive(range(6)))


def test_uses_calc_time():
    primitive = TimeSinceLast()
    primitive_h = TimeSinceLast(unit="hours")
    datetimes = pd.Series([datetime(2015, 6, 6), datetime(2015, 6, 7)])
    answer = 86400.0
    answer_h = 24.0
    assert answer == primitive(datetimes, time=datetime(2015, 6, 8))
    assert answer_h == primitive_h(datetimes, time=datetime(2015, 6, 8))


def test_call_multiple_args():
    primitive = Haversine()
    data1 = [(42.4, -71.1), (40.0, -122.4)]
    data2 = [(40.0, -122.4), (41.2, -96.75)]
    answer = [2631.231, 1343.289]

    for _ in range(2):
        assert np.round(primitive(data1, data2), 3).tolist() == answer


def test_get_function_called_once():
    class TestPrimitive(TransformPrimitive):
        def __init__(self):
            self.get_function_call_count = 0

        def get_function(self):
            self.get_function_call_count += 1

            def test(x):
                return x

            return test

    primitive = TestPrimitive()

    for _ in range(2):
        primitive(range(6))

    assert primitive.get_function_call_count == 1


def test_multiple_arg_string():
    class Primitive(TransformPrimitive):
        def __init__(self, bool=True, int=0, float=None):
            self.bool = bool
            self.int = int
            self.float = float

    primitive = Primitive(bool=True, int=4, float=0.1)
    string = primitive.get_args_string()
    assert string == ", int=4, float=0.1"


def test_single_args_string():
    assert IsIn([1, 2, 3]).get_args_string() == ", list_of_outputs=[1, 2, 3]"


def test_args_string_default():
    assert IsIn().get_args_string() == ""


def test_args_string_mixed():
    class Primitive(TransformPrimitive):
        def __init__(self, bool=True, int=0, float=None):
            self.bool = bool
            self.int = int
            self.float = float

    primitive = Primitive(bool=False, int=0)
    string = primitive.get_args_string()
    assert string == ", bool=False"


def test_args_string_undefined():
    string = Max().get_args_string()
    assert string == ""


def test_args_string_error():
    class Primitive(TransformPrimitive):
        def __init__(self, bool=True, int=0, float=None):
            pass

    with raises(AssertionError, match="must be attribute"):
        Primitive(bool=True, int=4, float=0.1).get_args_string()
