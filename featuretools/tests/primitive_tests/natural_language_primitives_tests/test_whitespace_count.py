import numpy as np
import pandas as pd

from featuretools.primitives import WhitespaceCount
from featuretools.tests.primitive_tests.utils import (
    PrimitiveTestBase,
    find_applicable_primitives,
    valid_dfs,
)


class TestWhitespaceCount(PrimitiveTestBase):
    primitive = WhitespaceCount

    def compare(self, primitive_initiated, test_cases, answers):
        primitive_func = primitive_initiated.get_function()
        primitive_answers = primitive_func(test_cases)
        return np.testing.assert_array_equal(answers, primitive_answers)

    def test_strings(self):
        x = pd.Series(
            ["", "hi im ethan!", "consecutive.    spaces.", " spaces-on-ends "],
        )
        answers = [0, 2, 4, 2]
        self.compare(self.primitive(), x, answers)

    def test_nan(self):
        x = pd.Series([np.nan, None, pd.NA, "", "This IS a STRING."])
        answers = [np.nan, np.nan, np.nan, 0, 3]
        self.compare(self.primitive(), x, answers)

    def test_with_featuretools(self, es):
        transform, aggregation = find_applicable_primitives(self.primitive)
        primitive_instance = self.primitive()
        transform.append(primitive_instance)
        valid_dfs(es, aggregation, transform, self.primitive)
