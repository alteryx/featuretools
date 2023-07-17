import numpy as np
import pandas as pd

from featuretools.primitives import NumConsecutiveGreaterMean, NumConsecutiveLessMean


class TestNumConsecutiveGreaterMean:
    primitive = NumConsecutiveGreaterMean

    def test_continuous_range(self):
        x = pd.Series(range(10))
        longest_sequence = [5, 6, 7, 8, 9]
        primitive_instance = self.primitive()
        primitive_func = primitive_instance.get_function()
        assert primitive_func(x) == len(longest_sequence)

    def test_subsequence_in_middle(self):
        x = pd.Series(
            [
                0.6,
                0.18,
                1.11,
                -0.19,
                0.25,
                -1.41,
                0.54,
                0.29,
                -1.59,
                1.67,
                1.19,
                0.44,
                2.39,
                -1.38,
                0.15,
                -1.16,
                1.54,
                -0.34,
                -1.41,
                0.58,
            ],
        )
        longest_sequence = [1.67, 1.19, 0.44, 2.39]
        primitive_instance = self.primitive()
        primitive_func = primitive_instance.get_function()
        assert primitive_func(x) == len(longest_sequence)

    def test_subsequence_at_start(self):
        x = pd.Series(
            [
                1.67,
                1.19,
                0.44,
                2.39,
                -0.19,
                0.6,
                0.18,
                1.11,
                0.25,
                -1.41,
                0.54,
                0.29,
                -1.59,
                -1.38,
                0.15,
                -1.16,
                1.54,
                -0.34,
                -1.41,
                0.58,
            ],
        )
        longest_sequence = [1.67, 1.19, 0.44, 2.39]
        primitive_instance = self.primitive()
        primitive_func = primitive_instance.get_function()
        assert primitive_func(x) == len(longest_sequence)

    def test_subsequence_at_end(self):
        x = pd.Series(
            [
                0.6,
                0.18,
                1.11,
                -0.19,
                0.25,
                -1.41,
                0.54,
                0.29,
                -1.59,
                -1.38,
                0.15,
                -1.16,
                1.54,
                -0.34,
                0.58,
                -1.41,
                1.67,
                1.19,
                0.44,
                2.39,
            ],
        )
        longest_sequence = [1.67, 1.19, 0.44, 2.39]
        primitive_instance = self.primitive()
        primitive_func = primitive_instance.get_function()
        assert primitive_func(x) == len(longest_sequence)

    def test_nan(self):
        x = pd.Series(range(10))
        x = pd.concat([x, pd.Series([np.nan] * 20)])
        longest_sequence = [5, 6, 7, 8, 9]

        # test ignoring NaN values
        primitive_instance = self.primitive()
        primitive_func = primitive_instance.get_function()
        assert primitive_func(x) == len(longest_sequence)

        # test skipna=False
        primitive_instance = self.primitive(skipna=False)
        primitive_func = primitive_instance.get_function()
        assert np.isnan(primitive_func(x))

    def test_inf(self):
        primitive_instance = self.primitive()
        primitive_func = primitive_instance.get_function()

        x = pd.Series(range(10))
        x = pd.concat([x, pd.Series([np.inf])])
        assert primitive_func(x) == 0

        x = pd.Series(range(10))
        x = pd.concat([x, pd.Series([np.NINF])])
        assert primitive_func(x) == 10

        x = pd.Series(range(10))
        x = pd.concat([x, pd.Series([np.NINF, np.inf, np.inf])])
        assert np.isnan(primitive_func(x))


class TestNumConsecutiveLessMean:
    primitive = NumConsecutiveLessMean

    def test_continuous_range(self):
        x = pd.Series(range(10))
        longest_sequence = [0, 1, 2, 3, 4]
        primitive_instance = self.primitive()
        primitive_func = primitive_instance.get_function()
        assert primitive_func(x) == len(longest_sequence)

    def test_subsequence_in_middle(self):
        x = pd.Series(
            [
                0.6,
                0.18,
                1.11,
                -0.19,
                0.25,
                -1.41,
                0.54,
                0.29,
                -1.59,
                1.67,
                1.19,
                0.44,
                2.39,
                -1.38,
                0.15,
                -1.16,
                1.54,
                -0.34,
                -1.41,
                0.58,
            ],
        )
        longest_sequence = [-1.38, 0.15, -1.16]
        primitive_instance = self.primitive()
        primitive_func = primitive_instance.get_function()
        assert primitive_func(x) == len(longest_sequence)

    def test_subsequence_at_start(self):
        x = pd.Series(
            [
                -1.38,
                0.15,
                -1.16,
                0.6,
                0.18,
                1.11,
                -0.19,
                0.25,
                -1.41,
                0.54,
                0.29,
                -1.59,
                1.67,
                1.19,
                0.44,
                2.39,
                1.54,
                -0.34,
                -1.41,
                0.58,
            ],
        )
        longest_sequence = [-1.38, 0.15, -1.16]
        primitive_instance = self.primitive()
        primitive_func = primitive_instance.get_function()
        assert primitive_func(x) == len(longest_sequence)

    def test_subsequence_at_end(self):
        x = pd.Series(
            [
                0.6,
                0.18,
                1.11,
                -0.19,
                0.25,
                -1.41,
                0.54,
                0.29,
                -1.59,
                1.67,
                1.19,
                0.44,
                2.39,
                1.54,
                -0.34,
                -1.41,
                0.58,
                -1.38,
                0.15,
                -1.16,
            ],
        )
        longest_sequence = [-1.38, 0.15, -1.16]
        primitive_instance = self.primitive()
        primitive_func = primitive_instance.get_function()
        assert primitive_func(x) == len(longest_sequence)

    def test_nan(self):
        x = pd.Series(range(10))
        x = pd.concat([x, pd.Series([np.nan] * 20)])
        longest_sequence = [0, 1, 2, 3, 4]

        # test ignoring NaN values
        primitive_instance = self.primitive()
        primitive_func = primitive_instance.get_function()
        assert primitive_func(x) == len(longest_sequence)

        # test skipna=False
        primitive_instance = self.primitive(skipna=False)
        primitive_func = primitive_instance.get_function()
        assert np.isnan(primitive_func(x))

    def test_inf(self):
        primitive_instance = self.primitive()
        primitive_func = primitive_instance.get_function()

        x = pd.Series(range(10))
        x = pd.concat([x, pd.Series([np.inf])])
        assert primitive_func(x) == 10

        x = pd.Series(range(10))
        x = pd.concat([x, pd.Series([np.NINF])])
        assert primitive_func(x) == 0

        x = pd.Series(range(10))
        x = pd.concat([x, pd.Series([np.NINF, np.inf, np.inf])])
        assert np.isnan(primitive_func(x))
