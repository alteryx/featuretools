import numpy as np

from featuretools.primitives.base import (
    AggregationPrimitive,
    PrimitiveBase,
    TransformPrimitive
)


class PrimitiveBaseTest(PrimitiveBase):

    def __init__(self, a, b, c=5, d=np.nan):
        self.a = a
        self.b = b
        self.c = c
        self.d = d


def test_get_args_string():
    example = PrimitiveBaseTest(1, 2, 3, 4).get_args_string()
    assert example == "a=1, b=2, c=3, d=4"

    example = PrimitiveBaseTest(1, 2, c=3, d=4).get_args_string()
    assert example == "a=1, b=2, c=3, d=4"

    example = PrimitiveBaseTest(1, 2, c=5, d=np.nan).get_args_string()
    assert example == "a=1, b=2"

    example = PrimitiveBaseTest(1, 2).get_args_string()
    assert example == "a=1, b=2"

    example = PrimitiveBaseTest(1, 2, c=np.nan, d=np.nan).get_args_string()
    assert example == "a=1, b=2, c=nan"


class PrimitiveBaseTest2(PrimitiveBase):
    def __init__(self):
        pass


def test_empty_args_string():
    assert PrimitiveBaseTest2().get_args_string() == ''


# We use transform primitive in primitive base as args_string is
# used in generate_name which is defined in transform and agg
class TransformBaseTest(TransformPrimitive):
    name = "TestTransformPrimitive"

    def __init__(self, a, b, c=0, d=1):
        self.a = a
        self.b = b
        self.c = c
        self.d = d


def test_generate_trans_name():

    name = "TestTransformPrimitive".upper()
    name += "(base_feature1, base_feature2, base_feature3, "
    name += "a=1, b=1, c=1)"
    base_features = ["base_feature" + str(i + 1) for i in range(3)]
    assert TransformBaseTest(1, 1, 1, 1).generate_name(base_features) == name


# We use transform primitive in primitive base as args_string is
# used in generate_name which is defined in transform and agg
class AggBaseTest(AggregationPrimitive):
    name = "TestAggPrimitive"

    def __init__(self, a, b, c=0, d=1):
        self.a = a
        self.b = b
        self.c = c
        self.d = d


def test_generate_agg_name():

    name = "TestAggPrimitive".upper()
    name += "(a.base_feature1, base_feature2, base_feature3cd, "
    name += "a=1, b=1, c=1)"

    base_features = ["base_feature" + str(i + 1) for i in range(3)]
    primitive = AggBaseTest(1, 1, 1, 1)
    generated_name = primitive.generate_name(base_features, 'a', 'b', 'c', 'd')
    assert name == generated_name
