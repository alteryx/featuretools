import numpy as np
import pandas as pd
from pytest import raises

from featuretools.primitives import (
    CountAboveMean,
    CountGreaterThan,
    CountInsideNthSTD,
    CountInsideRange,
    CountLessThan,
    CountOutsideNthSTD,
    CountOutsideRange,
)
from featuretools.tests.primitive_tests.utils import PrimitiveTestBase


class TestCountAboveMean(PrimitiveTestBase):
    primitive = CountAboveMean

    def test_regular(self):
        data = pd.Series([1, 2, 3, 4, 5])
        expected = 2
        primitive_func = self.primitive().get_function()
        actual = primitive_func(data)
        assert expected == actual

        data = pd.Series([1, 2, 3.1, 4, 5])
        expected = 3
        primitive_func = self.primitive().get_function()
        actual = primitive_func(data)
        assert expected == actual

    def test_nan_without_ignore_nan(self):
        data = pd.Series([np.nan, 1, 2, 3, 4, 5, np.nan, np.nan])
        expected = np.nan

        primitive_func = self.primitive(skipna=False).get_function()
        actual = primitive_func(data)
        assert np.isnan(actual) == np.isnan(expected)

        data = pd.Series([np.nan])
        primitive_func = self.primitive(skipna=False).get_function()
        actual = primitive_func(data)
        assert np.isnan(actual) == np.isnan(expected)

    def test_nan_with_ignore_nan(self):
        data = pd.Series([np.nan, 1, 2, 3, 4, 5, np.nan, np.nan])
        expected = 2
        primitive_func = self.primitive(skipna=True).get_function()
        actual = primitive_func(data)
        assert expected == actual

        data = pd.Series([np.nan, 1, 2, 3.1, 4, 5, np.nan, np.nan])
        expected = 3
        primitive_func = self.primitive(skipna=True).get_function()
        actual = primitive_func(data)
        assert expected == actual

        data = pd.Series([np.nan])
        expected = np.nan
        primitive_func = self.primitive(skipna=True).get_function()
        actual = primitive_func(data)
        assert np.isnan(actual) == np.isnan(expected)

    def test_inf(self):
        data = pd.Series([np.NINF, 1, 2, 3, 4, 5])
        expected = 5
        primitive_func = self.primitive().get_function()
        actual = primitive_func(data)
        assert expected == actual

        data = pd.Series([1, 2, 3, 4, 5, np.inf])
        expected = 0
        primitive_func = self.primitive().get_function()
        actual = primitive_func(data)
        assert expected == actual

        data = pd.Series([np.NINF, 1, 2, 3, 4, 5, np.inf])
        expected = np.nan
        primitive_func = self.primitive().get_function()
        actual = primitive_func(data)
        assert np.isnan(actual) == np.isnan(expected)

        primitive_func = self.primitive(skipna=False).get_function()
        actual = primitive_func(data)
        assert np.isnan(actual) == np.isnan(expected)


class TestCountGreaterThan(PrimitiveTestBase):
    primitive = CountGreaterThan

    def compare_results(self, data, thresholds, results):
        for threshold, result in zip(thresholds, results):
            primitive = self.primitive(threshold=threshold)
            function = primitive.get_function()
            assert function(data) == result
            assert isinstance(function(data), np.int64)

    def test_regular(self):
        data = pd.Series([-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5])
        thresholds = pd.Series([-5, -2, 0, 2, 5])
        results = pd.Series([10, 7, 5, 3, 0])
        self.compare_results(data, thresholds, results)

    def test_edges(self):
        data = pd.Series([-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5])
        thresholds = pd.Series([np.inf, np.NINF, None, np.nan])
        results = pd.Series([0, len(data), 0, 0])
        self.compare_results(data, thresholds, results)

    def test_nans(self):
        data = pd.Series([-5, -4, -3, np.inf, np.NINF, np.nan, 1, 2, 3, 4, 5])
        thresholds = pd.Series([np.inf, np.NINF, None, 0, np.nan])
        results = pd.Series([0, 9, 0, 6, 0])
        self.compare_results(data, thresholds, results)


class TestCountInsideNthSTD:
    primitive = CountInsideNthSTD

    def test_normal_distribution(self):
        x = pd.Series(
            [
                -76.0,
                41.0,
                -43.0,
                -152.0,
                -89.0,
                28.0,
                49.0,
                298.0,
                -132.0,
                146.0,
                -107.0,
                -26.0,
                26.0,
                -81.0,
                116.0,
                -217.0,
                -102.0,
                144.0,
                120.0,
                -130.0,
            ],
        )

        first_outliers = [-152.0, 298.0, 146.0, 116.0, -217.0, 144.0, 120.0]
        primitive_instance = self.primitive(1)
        primitive_func = primitive_instance.get_function()
        assert primitive_func(x) == len(x) - len(first_outliers)

        second_outliers = [298.0]
        primitive_instance = self.primitive(2)
        primitive_func = primitive_instance.get_function()
        assert primitive_func(x) == len(x) - len(second_outliers)

    def test_poisson_distribution(self):
        x = pd.Series(
            [
                1,
                1,
                3,
                3,
                0,
                0,
                1,
                3,
                3,
                1,
                2,
                3,
                2,
                0,
                1,
                3,
                2,
                1,
                0,
                2,
            ],
        )

        first_outliers = [3, 3, 0, 0, 3, 3, 3, 0, 3, 0]
        primitive_instance = self.primitive(1)
        primitive_func = primitive_instance.get_function()
        assert primitive_func(x) == len(x) - len(first_outliers)

        second_outliers = []
        primitive_instance = self.primitive(2)
        primitive_func = primitive_instance.get_function()
        assert primitive_func(x) == len(x) - len(second_outliers)

    def test_nan(self):
        # test if function ignores nan values
        x = pd.Series(
            [
                -76.0,
                41.0,
                -43.0,
                -152.0,
                -89.0,
                28.0,
                49.0,
                298.0,
                -132.0,
                146.0,
                -107.0,
                -26.0,
                26.0,
                -81.0,
                116.0,
                -217.0,
                -102.0,
                144.0,
                120.0,
                -130.0,
            ],
        )
        x = pd.concat([x, pd.Series([np.nan] * 20)])
        first_outliers = [-152.0, 298.0, 146.0, 116.0, -217.0, 144.0, 120.0]
        primitive_instance = self.primitive(1)
        primitive_func = primitive_instance.get_function()
        assert primitive_func(x) == len(x) - len(first_outliers) - 20

        # test a series with all nan values
        x = pd.Series([np.nan] * 20)

        primitive_instance = self.primitive(1)
        primitive_func = primitive_instance.get_function()
        assert primitive_func(x) == 0

    def test_negative_n(self):
        with raises(ValueError):
            self.primitive(-1)


class TestCountInsideRange(PrimitiveTestBase):
    primitive = CountInsideRange

    def test_integer_range(self):
        # all integers from -100 to 100
        x = pd.Series(np.arange(-100, 101, 1))
        primitive_instance = self.primitive(-100, 100)
        primitive_func = primitive_instance.get_function()
        assert primitive_func(x) == 201

        primitive_instance = self.primitive(-50, 50)
        primitive_func = primitive_instance.get_function()
        assert primitive_func(x) == 101

        primitive_instance = self.primitive(1, 1)
        primitive_func = primitive_instance.get_function()
        assert primitive_func(x) == 1

    def test_float_range(self):
        x = pd.Series(np.linspace(-3, 3, 10))

        primitive_instance = self.primitive(-3, 3)
        primitive_func = primitive_instance.get_function()
        assert primitive_func(x) == 10

        primitive_instance = self.primitive(-0.34, 1.68)
        primitive_func = primitive_instance.get_function()
        assert primitive_func(x) == 4

        primitive_instance = self.primitive(-3, -3)
        primitive_func = primitive_instance.get_function()
        assert primitive_func(x) == 1

    def test_nan(self):
        x = pd.Series(np.linspace(-3, 3, 10))
        x = pd.concat([x, pd.Series([np.nan] * 20)])

        primitive_instance = self.primitive(-0.34, 1.68)
        primitive_func = primitive_instance.get_function()
        assert primitive_func(x) == 4

        primitive_instance = self.primitive(-3, 3, False)
        primitive_func = primitive_instance.get_function()
        assert np.isnan(primitive_func(x))

    def test_inf(self):
        x = pd.Series(np.linspace(-3, 3, 10))
        num_NINF = 20
        x = pd.concat([x, pd.Series([np.NINF] * num_NINF)])
        num_inf = 10
        x = pd.concat([x, pd.Series([np.inf] * num_inf)])

        primitive_instance = self.primitive(-3, 3)
        primitive_func = primitive_instance.get_function()
        assert primitive_func(x) == 10

        primitive_instance = self.primitive(np.NINF, 3)
        primitive_func = primitive_instance.get_function()
        assert primitive_func(x) == 10 + num_NINF

        primitive_instance = self.primitive(-3, np.inf)
        primitive_func = primitive_instance.get_function()
        assert primitive_func(x) == 10 + num_inf


class TestCountLessThan(PrimitiveTestBase):
    primitive = CountLessThan

    def compare_answers(self, data, thresholds, answers):
        for threshold, answer in zip(thresholds, answers):
            primitive = self.primitive(threshold=threshold)
            function = primitive.get_function()
            assert function(data) == answer
            assert isinstance(function(data), np.int64)

    def test_regular(self):
        data = pd.Series([-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5])
        thresholds = pd.Series([-5, -2, 0, 2, 5])
        answers = pd.Series([0, 3, 5, 7, 10])
        self.compare_answers(data, thresholds, answers)

    def test_edges(self):
        data = pd.Series([-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5])
        thresholds = pd.Series([np.inf, np.NINF, None, np.nan])
        answers = pd.Series([len(data), 0, 0, 0])
        self.compare_answers(data, thresholds, answers)

    def test_nans(self):
        data = pd.Series([-5, -4, -3, np.inf, np.NINF, np.nan, 1, 2, 3, 4, 5])
        thresholds = pd.Series([np.inf, np.NINF, None, 0, np.nan])
        answers = pd.Series([9, 0, 0, 4, 0])
        self.compare_answers(data, thresholds, answers)


class TestCountOutsideNthSTD(PrimitiveTestBase):
    primitive = CountOutsideNthSTD

    def test_normal_distribution(self):
        x = pd.Series(
            [
                10,
                386,
                479,
                627,
                20,
                523,
                482,
                483,
                542,
                699,
                535,
                617,
                577,
                471,
                615,
                583,
                441,
                562,
                563,
                527,
                453,
                530,
                433,
                541,
                585,
                704,
                443,
                569,
                430,
                637,
                331,
                511,
                552,
                496,
                484,
                566,
                554,
                472,
                335,
                440,
                579,
                341,
                545,
                615,
                548,
                604,
                439,
                556,
                442,
                461,
                624,
                611,
                444,
                578,
                405,
                487,
                490,
                496,
                398,
                512,
                422,
                455,
                449,
                432,
                607,
                679,
                434,
                597,
                639,
                565,
                415,
                486,
                668,
                414,
                665,
                763,
                557,
                304,
                404,
                454,
                689,
                610,
                483,
                441,
                657,
                590,
                492,
                476,
                437,
                483,
                529,
                363,
                711,
                543,
            ],
        )
        outliers = [10, 20, 763]
        primitive_instance = self.primitive(2)
        primitive_func = primitive_instance.get_function()
        assert primitive_func(x) == len(outliers)

    def test_poisson_distribution(self):
        x = pd.Series(
            [
                1,
                1,
                3,
                3,
                0,
                0,
                1,
                3,
                3,
                1,
                2,
                3,
                2,
                0,
                1,
                3,
                2,
                1,
                0,
                2,
            ],
        )

        primitive_instance = self.primitive(1)
        primitive_func = primitive_instance.get_function()
        assert primitive_func(x) == 10

        primitive_instance = self.primitive(2)
        primitive_func = primitive_instance.get_function()
        assert primitive_func(x) == 0

    def test_nan(self):
        # test if function ignores nan values
        x = pd.Series(
            [
                -76.0,
                41.0,
                -43.0,
                -152.0,
                -89.0,
                28.0,
                49.0,
                298.0,
                -132.0,
                146.0,
                -107.0,
                -26.0,
                26.0,
                -81.0,
                116.0,
                -217.0,
                -102.0,
                144.0,
                120.0,
                -130.0,
            ],
        )
        x = pd.concat([x, pd.Series([np.nan * 20])])
        primitive_instance = self.primitive(1)
        primitive_func = primitive_instance.get_function()
        assert primitive_func(x) == 7

        # test a series with all nan values
        x = pd.Series([np.nan] * 20)

        primitive_instance = self.primitive(1)
        primitive_func = primitive_instance.get_function()
        assert primitive_func(x) == 0

    def test_negative_n(self):
        with raises(ValueError):
            self.primitive(-1)


class TestCountOutsideRange(PrimitiveTestBase):
    primitive = CountOutsideRange

    def test_integer_range(self):
        # all integers from -100 to 100
        x = pd.Series(np.arange(-100, 101, 1))
        primitive_instance = CountOutsideRange(-100, 100)
        primitive_func = primitive_instance.get_function()
        assert primitive_func(x) == 0

        primitive_instance = CountOutsideRange(-50, 50)
        primitive_func = primitive_instance.get_function()
        assert primitive_func(x) == 100

        primitive_instance = CountOutsideRange(1, 1)
        primitive_func = primitive_instance.get_function()
        assert primitive_func(x) == len(x) - 1

    def test_float_range(self):
        x = pd.Series(np.linspace(-3, 3, 10))

        primitive_instance = CountOutsideRange(-3, 3)
        primitive_func = primitive_instance.get_function()
        assert primitive_func(x) == 0

        primitive_instance = CountOutsideRange(-0.34, 1.68)
        primitive_func = primitive_instance.get_function()
        assert primitive_func(x) == 6

        primitive_instance = CountOutsideRange(-3, -3)
        primitive_func = primitive_instance.get_function()
        assert primitive_func(x) == 9

    def test_nan(self):
        x = pd.Series(np.linspace(-3, 3, 10))
        x = pd.concat([x, pd.Series([np.nan] * 20)])
        primitive_instance = CountOutsideRange(-0.34, 1.68)
        primitive_func = primitive_instance.get_function()
        assert primitive_func(x) == 6

        primitive_instance = CountOutsideRange(-3, 3, False)
        primitive_func = primitive_instance.get_function()
        assert np.isnan(primitive_func(x))

    def test_inf(self):
        x = pd.Series(np.linspace(-3, 3, 10))
        num_NINF = 20
        x = pd.concat([x, pd.Series([np.NINF] * num_NINF)])
        num_inf = 10
        x = pd.concat([x, pd.Series([np.inf] * num_inf)])

        primitive_instance = CountOutsideRange(-3, 3)
        primitive_func = primitive_instance.get_function()
        assert primitive_func(x) == num_inf + num_NINF

        primitive_instance = CountOutsideRange(-0.34, 1.68)
        primitive_func = primitive_instance.get_function()
        assert primitive_func(x) == 6 + num_inf + num_NINF

        primitive_instance = CountOutsideRange(np.NINF, 3)
        primitive_func = primitive_instance.get_function()
        assert primitive_func(x) == num_inf

        primitive_instance = CountOutsideRange(-3, np.inf)
        primitive_func = primitive_instance.get_function()
        assert primitive_func(x) == num_NINF
