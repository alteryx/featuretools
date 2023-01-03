import numpy as np
import pandas as pd

from featuretools.primitives import SameAsPrevious


class TestSameAsPrevious:
    primitive = SameAsPrevious

    def test_ints(self):
        primitive_func = SameAsPrevious().get_function()
        array = pd.Series([1, 2, 2, 3, 2], dtype="int64")
        answer = primitive_func(array)
        correct_answer = pd.Series([False, False, True, False, False])
        pd.testing.assert_series_equal(answer, correct_answer)

    def test_ints(self):
        primitive_func = SameAsPrevious().get_function()
        array = pd.Series([1, 2, 2, 3, 2], dtype="Int64")
        answer = primitive_func(array)
        correct_answer = pd.Series([False, False, True, False, False], dtype="boolean")
        pd.testing.assert_series_equal(answer, correct_answer)

    def test_floats(self):
        primitive_func = SameAsPrevious().get_function()
        array = pd.Series([1.0, 2.5, 2.5, 3.0, 2.0], dtype="float64")
        answer = primitive_func(array)
        correct_answer = pd.Series([False, False, True, False, False])
        pd.testing.assert_series_equal(answer, correct_answer)

    def test_mixed(self):
        primitive_func = SameAsPrevious().get_function()
        array = pd.Series([1, 2, 2.0, 3, 2.0], dtype="float64")
        answer = primitive_func(array)
        correct_answer = pd.Series([False, False, True, False, False])
        np.testing.assert_array_equal(answer, correct_answer)

    def test_nan(self):
        primitive_instance = self.primitive()
        primitive_func = primitive_instance.get_function()
        array = pd.Series([1, np.nan, 3, np.nan, 2], dtype="float64")
        answer = primitive_func(array)
        correct_answer = pd.Series([False, True, False, True, False])
        np.testing.assert_array_equal(answer, correct_answer)

    def test_all_nan(self):
        primitive_instance = self.primitive()
        primitive_func = primitive_instance.get_function()
        array = pd.Series([np.nan, np.nan, np.nan, np.nan], dtype="float64")
        answer = primitive_func(array)
        correct_answer = pd.Series([False, False, False, False])
        np.testing.assert_array_equal(answer, correct_answer)

    def test_inf(self):
        primitive_instance = self.primitive()
        primitive_func = primitive_instance.get_function()
        array = pd.Series([1, np.inf, 3, np.inf, 2], dtype="float64")
        answer = primitive_func(array)
        correct_answer = pd.Series([False, False, False, False, False])
        np.testing.assert_array_equal(answer, correct_answer)

    def test_all_inf(self):
        primitive_instance = self.primitive()
        primitive_func = primitive_instance.get_function()
        array = pd.Series([np.inf, np.inf, np.inf, np.inf], dtype="float64")
        answer = primitive_func(array)
        correct_answer = pd.Series([False, True, True, True])
        np.testing.assert_array_equal(answer, correct_answer)

    def test_fill_method_bfill(self):
        primitive_instance = self.primitive(fill_method="bfill")
        primitive_func = primitive_instance.get_function()
        array = pd.Series([1, np.nan, 3, 2, 2], dtype="float64")
        answer = primitive_func(array)
        correct_answer = pd.Series([False, False, True, False, True])
        np.testing.assert_array_equal(answer, correct_answer)

    def test_fill_method_bfill_with_limit(self):
        primitive_instance = self.primitive(fill_method="bfill", limit=2)
        primitive_func = primitive_instance.get_function()
        array = pd.Series([1, np.nan, np.nan, np.nan, 2, 3], dtype="float64")
        answer = primitive_func(array)
        correct_answer = pd.Series([False, False, False, True, True, False])
        np.testing.assert_array_equal(answer, correct_answer)
