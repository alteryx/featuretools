import numpy as np
import pandas as pd
from pytest import raises

from featuretools.primitives import PercentChange
from featuretools.tests.primitive_tests.utils import (
    PrimitiveTestBase,
    find_applicable_primitives,
    valid_dfs,
)


class TestPercentChange(PrimitiveTestBase):
    primitive = PercentChange

    def test_regular(self):
        data = pd.Series([2, 5, 15, 3, 3, 9, 4.5])
        answer = pd.Series([np.nan, 1.5, 2.0, -0.8, 0, 2.0, -0.5])
        primtive_func = self.primitive().get_function()
        given_answer = primtive_func(data)
        np.testing.assert_array_equal(given_answer, answer)

    def test_raises(self):
        with raises(ValueError):
            self.primitive(fill_method="invalid")

    def test_period(self):
        data = pd.Series([2, 4, 8])
        answer = pd.Series([np.nan, np.nan, 3])
        primtive_func = self.primitive(periods=2).get_function()
        given_answer = primtive_func(data)
        np.testing.assert_array_equal(given_answer, answer)
        primtive_func = self.primitive(periods=2).get_function()
        data = pd.Series([2, 4, 8] + [np.nan] * 4)
        primtive_func = self.primitive(limit=2).get_function()
        answer = pd.Series([np.nan, 1, 1, 0, 0, np.nan, np.nan])
        given_answer = primtive_func(data)
        np.testing.assert_array_equal(given_answer, answer)

    def test_nan(self):
        data = pd.Series([np.nan, 5, 10, 20, np.nan, 10, np.nan])
        answer = pd.Series([np.nan, np.nan, 1, 1, 0, -0.5, 0])
        primtive_func = self.primitive().get_function()
        given_answer = primtive_func(data)
        np.testing.assert_array_equal(given_answer, answer)

    def test_zero(self):
        data = pd.Series([2, 0, 0, 5, 0, -4])
        answer = pd.Series([np.nan, -1, np.nan, np.inf, -1, np.NINF])
        primtive_func = self.primitive().get_function()
        given_answer = primtive_func(data)
        np.testing.assert_array_equal(given_answer, answer)

    def test_inf(self):
        data = pd.Series([0, np.inf, 0, 5, np.NINF, np.inf, np.NINF])
        answer = pd.Series([np.nan, np.inf, -1, np.inf, np.NINF, np.nan, np.nan])
        primtive_func = self.primitive().get_function()
        given_answer = primtive_func(data)
        np.testing.assert_array_equal(given_answer, answer)

    def test_freq(self):
        dates = pd.DatetimeIndex(
            ["2018-01-01", "2018-01-02", "2018-01-03", "2018-01-05"],
        )
        data = pd.Series([1, 2, 3, 4], index=dates)
        answer = pd.Series([np.nan, 1.0, 0.5, np.nan])
        date_offset = pd.tseries.offsets.DateOffset(days=1)
        primtive_func = self.primitive(freq=date_offset).get_function()
        given_answer = primtive_func(data)
        np.testing.assert_array_equal(given_answer, answer)

    def test_with_featuretools(self, pd_es):
        transform, aggregation = find_applicable_primitives(self.primitive)
        primitive_instantiate = self.primitive
        transform.append(primitive_instantiate)
        valid_dfs(pd_es, aggregation, transform, self.primitive)
