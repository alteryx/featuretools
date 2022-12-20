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
    primitive = MaxConsecutiveFalse

    def test_regular(self):
        primitive_instance = self.primitive()
        primitive_func = primitive_instance.get_function()
        array = pd.Series([False, False, False, True, True, False, True], dtype="bool")
        assert primitive_func(array) == 3

    def test_all_true(self):
        primitive_instance = self.primitive()
        primitive_func = primitive_instance.get_function()
        array = pd.Series([True, True, True, True], dtype="bool")
        assert primitive_func(array) == 0

    def test_all_false(self):
        primitive_instance = self.primitive()
        primitive_func = primitive_instance.get_function()
        array = pd.Series([False, False, False], dtype="bool")
        assert primitive_func(array) == 3


class TestMaxConsecutiveTrue:
    primitive = MaxConsecutiveTrue

    def test_regular(self):
        primitive_instance = self.primitive()
        primitive_func = primitive_instance.get_function()
        array = pd.Series([True, False, True, True, True, False, True], dtype="bool")
        assert primitive_func(array) == 3

    def test_all_true(self):
        primitive_instance = self.primitive()
        primitive_func = primitive_instance.get_function()
        array = pd.Series([True, True, True, True], dtype="bool")
        assert primitive_func(array) == 4

    def test_all_false(self):
        primitive_instance = self.primitive()
        primitive_func = primitive_instance.get_function()
        array = pd.Series([False, False, False], dtype="bool")
        assert primitive_func(array) == 0


class TestMaxConsecutiveNegatives:
    primitive = MaxConsecutiveNegatives

    def test_regular(self):
        primitive_instance = self.primitive()
        primitive_func = primitive_instance.get_function()
        array = pd.Series([1.3, -3.4, -1, -4, 10, -1.7, -4.9])
        assert primitive_func(array) == 3

    def test_all_int(self):
        primitive_instance = self.primitive()
        primitive_func = primitive_instance.get_function()
        array = pd.Series([1, -1, 2, 4, -5])
        assert primitive_func(array) == 1

    def test_all_float(self):
        primitive_instance = self.primitive()
        primitive_func = primitive_instance.get_function()
        array = pd.Series([1.0, -1.0, -2.0, 0.0, 5.0])
        assert primitive_func(array) == 2

    def test_with_nan(self):
        primitive_instance = self.primitive()
        primitive_func = primitive_instance.get_function()
        array = pd.Series([1, pd.NA, -2, -3])
        assert primitive_func(array) == 2

    def test_with_nan_skipna(self):
        primitive_instance = self.primitive(skipna=False)
        primitive_func = primitive_instance.get_function()
        array = pd.Series([-1, pd.NA, -2, -3])
        assert primitive_func(array) == 2

    def test_all_nan(self):
        primitive_instance = self.primitive()
        primitive_func = primitive_instance.get_function()
        array = pd.Series([pd.NA, pd.NA, pd.NA, pd.NA])
        assert np.isnan(primitive_func(array))

    def test_all_nan_skipna(self):
        primitive_instance = self.primitive(skipna=True)
        primitive_func = primitive_instance.get_function()
        array = pd.Series([pd.NA, pd.NA, pd.NA, pd.NA])
        assert np.isnan(primitive_func(array))


class TestMaxConsecutivePositives:
    primitive = MaxConsecutivePositives

    def test_regular(self):
        primitive_instance = self.primitive()
        primitive_func = primitive_instance.get_function()
        array = pd.Series([1.3, -3.4, 1, 4, 10, -1.7, -4.9])
        assert primitive_func(array) == 3

    def test_all_int(self):
        primitive_instance = self.primitive()
        primitive_func = primitive_instance.get_function()
        array = pd.Series([1, -1, 2, 4, -5])
        assert primitive_func(array) == 2

    def test_all_float(self):
        primitive_instance = self.primitive()
        primitive_func = primitive_instance.get_function()
        array = pd.Series([1.0, -1.0, 2.0, 4.0, 5.0])
        assert primitive_func(array) == 3

    def test_with_nan(self):
        primitive_instance = self.primitive()
        primitive_func = primitive_instance.get_function()
        array = pd.Series([1, np.nan, 2, -3])
        assert primitive_func(array) == 2

    def test_with_nan_skipna(self):
        primitive_instance = self.primitive(skipna=False)
        primitive_func = primitive_instance.get_function()
        array = pd.Series([1, np.nan, 2, -3])
        assert primitive_func(array) == 1

    def test_all_nan(self):
        primitive_instance = self.primitive()
        primitive_func = primitive_instance.get_function()
        array = pd.Series([np.nan, np.nan, np.nan, np.nan])
        assert np.isnan(primitive_func(array))

    def test_all_nan_skipna(self):
        primitive_instance = self.primitive(skipna=True)
        primitive_func = primitive_instance.get_function()
        array = pd.Series([np.nan, np.nan, np.nan, np.nan])
        assert np.isnan(primitive_func(array))


class TestMaxConsecutiveZeros:
    primitive = MaxConsecutiveZeros

    def test_regular(self):
        primitive_instance = self.primitive()
        primitive_func = primitive_instance.get_function()
        array = pd.Series([1.3, -3.4, 0, 0, 0.0, 1.7, -4.9])
        assert primitive_func(array) == 3

    def test_all_int(self):
        primitive_instance = self.primitive()
        primitive_func = primitive_instance.get_function()
        array = pd.Series([1, -1, 0, 0, -5])
        assert primitive_func(array) == 2

    def test_all_float(self):
        primitive_instance = self.primitive()
        primitive_func = primitive_instance.get_function()
        array = pd.Series([1.0, 0.0, 0.0, 0.0, -5.3])
        assert primitive_func(array) == 3

    def test_with_nan(self):
        primitive_instance = self.primitive()
        primitive_func = primitive_instance.get_function()
        array = pd.Series([0, np.nan, 0, -3])
        assert primitive_func(array) == 2

    def test_with_nan_skipna(self):
        primitive_instance = self.primitive(skipna=False)
        primitive_func = primitive_instance.get_function()
        array = pd.Series([0, np.nan, 0, -3])
        assert primitive_func(array) == 1

    def test_all_nan(self):
        primitive_instance = self.primitive()
        primitive_func = primitive_instance.get_function()
        array = pd.Series([np.nan, np.nan, np.nan, np.nan])
        assert np.isnan(primitive_func(array))

    def test_all_nan_skipna(self):
        primitive_instance = self.primitive(skipna=True)
        primitive_func = primitive_instance.get_function()
        array = pd.Series([np.nan, np.nan, np.nan, np.nan])
        assert np.isnan(primitive_func(array))
