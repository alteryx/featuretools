import numpy as np
import pandas as pd

from featuretools.primitives import UpperCaseCount
from featuretools.tests.primitive_tests.utils import (
    PrimitiveTestBase,
    find_applicable_primitives,
    valid_dfs,
)


class TestUpperCaseCount(PrimitiveTestBase):
    primitive = UpperCaseCount

    def test_strings(self):
        x = pd.Series(
            ["This IS a STRING.", "Testing AaA", "Testing AAA-BBB", "testing aaa"],
        )
        primitive_func = self.primitive().get_function()
        answers = [9.0, 3.0, 7.0, 0.0]
        np.testing.assert_array_equal(primitive_func(x), answers)

    def test_nan(self):
        x = pd.Series([np.nan, "", "This IS a STRING."])
        primitive_func = self.primitive().get_function()
        answers = [np.nan, 0.0, 9.0]
        np.testing.assert_array_equal(primitive_func(x), answers)

    def test_with_featuretools(self, es):
        transform, aggregation = find_applicable_primitives(self.primitive)
        primitive_instance = self.primitive()
        transform.append(primitive_instance)
        valid_dfs(es, aggregation, transform, self.primitive)
