import numpy as np

from featuretools.primitives.base import PrimitiveBase


class PrimitiveBaseTest(PrimitiveBase):

    def __init__(self, a, b, c=5, d=np.nan):
        self.a = a
        self.b = b
        self.c = c
        self.d = d


def test_get_args_string():
    example = PrimitiveBaseTest(1, 2, 3, 4).get_args_string()
    assert example == "a=1,b=2,c=3,d=4"

    example = PrimitiveBaseTest(1, 2, c=3, d=4).get_args_string()
    assert example == "a=1,b=2,c=3,d=4"

    example = PrimitiveBaseTest(1, 2, c=5, d=np.nan).get_args_string()
    assert example == "a=1,b=2"

    example = PrimitiveBaseTest(1, 2).get_args_string()
    assert example == "a=1,b=2"

    example = PrimitiveBaseTest(1, 2, c=np.nan, d=np.nan).get_args_string()
    assert example == "a=1,b=2,c=nan"
