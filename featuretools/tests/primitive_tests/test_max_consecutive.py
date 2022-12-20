import numpy as np
import pandas as pd
import pytest

from featuretools.primitives import (
    MaxConsecutiveFalse,
    MaxConsecutiveNegatives,
    MaxConsecutivePositives,
    MaxConsecutiveTrue,
    MaxConsecutiveZeros,
)


class TestMaxConsecutiveFalse:
    def test_regular(self):
        primitive_instance = MaxConsecutiveFalse()
        primitive_func = primitive_instance.get_function()
        array = pd.Series([False, False, False, True, True, False, True], dtype="bool")
        assert primitive_func(array) == 3

    def test_all_true(self):
        primitive_instance = MaxConsecutiveFalse()
        primitive_func = primitive_instance.get_function()
        array = pd.Series([True, True, True, True], dtype="bool")
        assert primitive_func(array) == 0

    def test_all_false(self):
        primitive_instance = MaxConsecutiveFalse()
        primitive_func = primitive_instance.get_function()
        array = pd.Series([False, False, False], dtype="bool")
        assert primitive_func(array) == 3


class TestMaxConsecutiveTrue:
    def test_regular(self):
        primitive_instance = MaxConsecutiveTrue()
        primitive_func = primitive_instance.get_function()
        array = pd.Series([True, False, True, True, True, False, True], dtype="bool")
        assert primitive_func(array) == 3

    def test_all_true(self):
        primitive_instance = MaxConsecutiveTrue()
        primitive_func = primitive_instance.get_function()
        array = pd.Series([True, True, True, True], dtype="bool")
        assert primitive_func(array) == 4

    def test_all_false(self):
        primitive_instance = MaxConsecutiveTrue()
        primitive_func = primitive_instance.get_function()
        array = pd.Series([False, False, False], dtype="bool")
        assert primitive_func(array) == 0


@pytest.mark.parametrize("dtype", ["float64", "int64"])
class TestMaxConsecutiveNegatives:
    def test_regular(self, dtype):
        if dtype == "int64":
            pytest.skip("test array contains floats which are not supported int64")
        primitive_instance = MaxConsecutiveNegatives()
        primitive_func = primitive_instance.get_function()
        array = pd.Series([1.3, -3.4, -1, -4, 10, -1.7, -4.9], dtype=dtype)
        assert primitive_func(array) == 3

    def test_all_int(self, dtype):
        primitive_instance = MaxConsecutiveNegatives()
        primitive_func = primitive_instance.get_function()
        array = pd.Series([1, -1, 2, 4, -5], dtype=dtype)
        assert primitive_func(array) == 1

    def test_all_float(self, dtype):
        if dtype == "int64":
            pytest.skip("test array contains floats which are not supported int64")
        primitive_instance = MaxConsecutiveNegatives()
        primitive_func = primitive_instance.get_function()
        array = pd.Series([1.0, -1.0, -2.0, 0.0, 5.0], dtype=dtype)
        assert primitive_func(array) == 2

    def test_with_nan(self, dtype):
        if dtype == "int64":
            pytest.skip("nans not supported in int64")
        primitive_instance = MaxConsecutiveNegatives()
        primitive_func = primitive_instance.get_function()
        array = pd.Series([1, np.nan, -2, -3], dtype=dtype)
        assert primitive_func(array) == 2

    def test_with_nan_skipna(self, dtype):
        if dtype == "int64":
            pytest.skip("nans not supported in int64")
        primitive_instance = MaxConsecutiveNegatives(skipna=False)
        primitive_func = primitive_instance.get_function()
        array = pd.Series([-1, np.nan, -2, -3], dtype=dtype)
        assert primitive_func(array) == 2

    def test_all_nan(self, dtype):
        if dtype == "int64":
            pytest.skip("nans not supported in int64")
        primitive_instance = MaxConsecutiveNegatives()
        primitive_func = primitive_instance.get_function()
        array = pd.Series([np.nan, np.nan, np.nan, np.nan], dtype=dtype)
        assert np.isnan(primitive_func(array))

    def test_all_nan_skipna(self, dtype):
        if dtype == "int64":
            pytest.skip("nans not supported in int64")
        primitive_instance = MaxConsecutiveNegatives(skipna=True)
        primitive_func = primitive_instance.get_function()
        array = pd.Series([np.nan, np.nan, np.nan, np.nan], dtype=dtype)
        assert np.isnan(primitive_func(array))


@pytest.mark.parametrize("dtype", ["float64", "int64"])
class TestMaxConsecutivePositives:
    def test_regular(self, dtype):
        if dtype == "int64":
            pytest.skip("test array contains floats which are not supported int64")
        primitive_instance = MaxConsecutivePositives()
        primitive_func = primitive_instance.get_function()
        array = pd.Series([1.3, -3.4, 1, 4, 10, -1.7, -4.9], dtype=dtype)
        assert primitive_func(array) == 3

    def test_all_int(self, dtype):
        primitive_instance = MaxConsecutivePositives()
        primitive_func = primitive_instance.get_function()
        array = pd.Series([1, -1, 2, 4, -5], dtype=dtype)
        assert primitive_func(array) == 2

    def test_all_float(self, dtype):
        if dtype == "int64":
            pytest.skip("test array contains floats which are not supported int64")
        primitive_instance = MaxConsecutivePositives()
        primitive_func = primitive_instance.get_function()
        array = pd.Series([1.0, -1.0, 2.0, 4.0, 5.0], dtype=dtype)
        assert primitive_func(array) == 3

    def test_with_nan(self, dtype):
        if dtype == "int64":
            pytest.skip("nans not supported in int64")
        primitive_instance = MaxConsecutivePositives()
        primitive_func = primitive_instance.get_function()
        array = pd.Series([1, np.nan, 2, -3], dtype=dtype)
        assert primitive_func(array) == 2

    def test_with_nan_skipna(self, dtype):
        if dtype == "int64":
            pytest.skip("nans not supported in int64")
        primitive_instance = MaxConsecutivePositives(skipna=False)
        primitive_func = primitive_instance.get_function()
        array = pd.Series([1, np.nan, 2, -3], dtype=dtype)
        assert primitive_func(array) == 1

    def test_all_nan(self, dtype):
        if dtype == "int64":
            pytest.skip("nans not supported in int64")
        primitive_instance = MaxConsecutivePositives()
        primitive_func = primitive_instance.get_function()
        array = pd.Series([np.nan, np.nan, np.nan, np.nan], dtype=dtype)
        assert np.isnan(primitive_func(array))

    def test_all_nan_skipna(self, dtype):
        if dtype == "int64":
            pytest.skip("nans not supported in int64")
        primitive_instance = MaxConsecutivePositives(skipna=True)
        primitive_func = primitive_instance.get_function()
        array = pd.Series([np.nan, np.nan, np.nan, np.nan], dtype=dtype)
        assert np.isnan(primitive_func(array))


@pytest.mark.parametrize("dtype", ["float64", "int64"])
class TestMaxConsecutiveZeros:
    def test_regular(self, dtype):
        if dtype == "int64":
            pytest.skip("test array contains floats which are not supported int64")
        primitive_instance = MaxConsecutiveZeros()
        primitive_func = primitive_instance.get_function()
        array = pd.Series([1.3, -3.4, 0, 0, 0.0, 1.7, -4.9], dtype=dtype)
        assert primitive_func(array) == 3

    def test_all_int(self, dtype):
        primitive_instance = MaxConsecutiveZeros()
        primitive_func = primitive_instance.get_function()
        array = pd.Series([1, -1, 0, 0, -5], dtype=dtype)
        assert primitive_func(array) == 2

    def test_all_float(self, dtype):
        if dtype == "int64":
            pytest.skip("test array contains floats which are not supported int64")
        primitive_instance = MaxConsecutiveZeros()
        primitive_func = primitive_instance.get_function()
        array = pd.Series([1.0, 0.0, 0.0, 0.0, -5.3], dtype=dtype)
        assert primitive_func(array) == 3

    def test_with_nan(self, dtype):
        if dtype == "int64":
            pytest.skip("nans not supported in int64")
        primitive_instance = MaxConsecutiveZeros()
        primitive_func = primitive_instance.get_function()
        array = pd.Series([0, np.nan, 0, -3], dtype=dtype)
        assert primitive_func(array) == 2

    def test_with_nan_skipna(self, dtype):
        if dtype == "int64":
            pytest.skip("nans not supported in int64")
        primitive_instance = MaxConsecutiveZeros(skipna=False)
        primitive_func = primitive_instance.get_function()
        array = pd.Series([0, np.nan, 0, -3], dtype=dtype)
        assert primitive_func(array) == 1

    def test_all_nan(self, dtype):
        if dtype == "int64":
            pytest.skip("nans not supported in int64")
        primitive_instance = MaxConsecutiveZeros()
        primitive_func = primitive_instance.get_function()
        array = pd.Series([np.nan, np.nan, np.nan, np.nan], dtype=dtype)
        assert np.isnan(primitive_func(array))

    def test_all_nan_skipna(self, dtype):
        if dtype == "int64":
            pytest.skip("nans not supported in int64")
        primitive_instance = MaxConsecutiveZeros(skipna=True)
        primitive_func = primitive_instance.get_function()
        array = pd.Series([np.nan, np.nan, np.nan, np.nan], dtype=dtype)
        assert np.isnan(primitive_func(array))
