import numpy as np
import pandas as pd
import pytest

from featuretools.primitives import AbsoluteDiff


class TestAbsoluteDiff:
    def test_nan(self):
        data = pd.Series([np.nan, 5, 10, 20, np.nan, 10, np.nan])
        answer = pd.Series([np.nan, np.nan, 5, 10, 0, 10, 0])
        primitive_func = AbsoluteDiff().get_function()
        given_answer = primitive_func(data)
        np.testing.assert_array_equal(given_answer, answer)

    def test_regular(self):
        data = pd.Series([2, 5, 15, 3, 9, 4.5])
        answer = pd.Series([np.nan, 3, 10, 12, 6, 4.5])
        primitive_func = AbsoluteDiff().get_function()
        given_answer = primitive_func(data)
        np.testing.assert_array_equal(given_answer, answer)

    def test_method(self):
        data = pd.Series([2, np.nan, 15, 3, np.nan, 4.5])
        answer = pd.Series([np.nan, 13, 0, 12, 1.5, 0])
        primitive_func = AbsoluteDiff(method="backfill").get_function()
        given_answer = primitive_func(data)
        np.testing.assert_array_equal(given_answer, answer)

    def test_limit(self):
        data = pd.Series([2, np.nan, np.nan, np.nan, 3.0, 4.5])
        answer = pd.Series([np.nan, 0, 0, np.nan, np.nan, 1.5])
        primitive_func = AbsoluteDiff(limit=2).get_function()
        given_answer = primitive_func(data)
        np.testing.assert_array_equal(given_answer, answer)

    def test_zero(self):
        data = pd.Series([2, 0, 0, 5, 0, -4])
        answer = pd.Series([np.nan, 2, 0, 5, 5, 4])
        primitive_func = AbsoluteDiff().get_function()
        given_answer = primitive_func(data)
        np.testing.assert_array_equal(given_answer, answer)

    def test_empty(self):
        data = pd.Series([], dtype="float64")
        answer = pd.Series([], dtype="float64")
        primitive_func = AbsoluteDiff().get_function()
        given_answer = primitive_func(data)
        np.testing.assert_array_equal(given_answer, answer)

    def test_inf(self):
        data = pd.Series([0, np.inf, 0, 5, np.NINF, np.inf, np.NINF])
        answer = pd.Series([np.nan, np.inf, np.inf, 5, np.inf, np.inf, np.inf])
        primitive_func = AbsoluteDiff().get_function()
        given_answer = primitive_func(data)
        np.testing.assert_array_equal(given_answer, answer)

    def test_raises(self):
        with pytest.raises(ValueError):
            AbsoluteDiff(method="invalid")
