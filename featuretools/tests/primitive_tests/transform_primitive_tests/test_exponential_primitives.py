import numpy as np
import pandas as pd

from featuretools.primitives import (
    ExponentialWeightedAverage,
    ExponentialWeightedSTD,
    ExponentialWeightedVariance,
)


def test_regular_com_avg():
    primitive_instance = ExponentialWeightedAverage(com=0.5)
    primitive_func = primitive_instance.get_function()
    array = pd.Series([1, 2, 7, 5])
    answer = pd.Series(primitive_func(array))
    correct_answer = pd.Series([1.0, 1.75, 5.384615384615384, 5.125])
    pd.testing.assert_series_equal(answer, correct_answer)


def test_regular_span_avg():
    primitive_instance = ExponentialWeightedAverage(span=1.5)
    primitive_func = primitive_instance.get_function()
    array = pd.Series([1, 2, 7, 5])
    answer = pd.Series(primitive_func(array))
    correct_answer = pd.Series([1.0, 1.8333333333333335, 6.0, 5.198717948717948])
    pd.testing.assert_series_equal(answer, correct_answer)


def test_regular_halflife_avg():
    primitive_instance = ExponentialWeightedAverage(halflife=2.7)
    primitive_func = primitive_instance.get_function()
    array = pd.Series([1, 2, 7, 5])
    answer = pd.Series(primitive_func(array))
    correct_answer = pd.Series(
        [1.0, 1.563830114594977, 3.8556233149044865, 4.2592901785684205],
    )
    pd.testing.assert_series_equal(answer, correct_answer)


def test_regular_alpha_avg():
    primitive_instance = ExponentialWeightedAverage(alpha=0.8)
    primitive_func = primitive_instance.get_function()
    array = pd.Series([1, 2, 7, 5])
    answer = pd.Series(primitive_func(array))
    correct_answer = pd.Series([1.0, 1.8333333333333335, 6.0, 5.198717948717948])
    pd.testing.assert_series_equal(answer, correct_answer)


def test_na_avg():
    primitive_instance = ExponentialWeightedAverage(com=0.5)
    primitive_func = primitive_instance.get_function()
    array = pd.Series([1, 2, 7, np.nan, 5])
    answer = pd.Series(primitive_func(array))
    correct_answer = pd.Series(
        [1.0, 1.75, 5.384615384615384, 5.384615384615384, 5.053191489361702],
    )
    pd.testing.assert_series_equal(answer, correct_answer)


def test_ignorena_true_avg():
    primitive_instance = ExponentialWeightedAverage(com=0.5, ignore_na=True)
    primitive_func = primitive_instance.get_function()
    array = pd.Series([1, 2, 7, np.nan, 5])
    answer = pd.Series(primitive_func(array))
    correct_answer = pd.Series(
        [1.0, 1.75, 5.384615384615384, 5.384615384615384, 5.125],
    )
    pd.testing.assert_series_equal(answer, correct_answer)


def test_regular_com_std():
    primitive_instance = ExponentialWeightedSTD(com=0.5)
    primitive_func = primitive_instance.get_function()
    array = pd.Series([1, 2, 7, 5])
    answer = pd.Series(primitive_func(array))
    correct_answer = pd.Series(
        [np.nan, 0.7071067811865475, 3.584153156068229, 2.0048019276803304],
    )
    pd.testing.assert_series_equal(answer, correct_answer)


def test_regular_span_std():
    primitive_instance = ExponentialWeightedSTD(span=1.5)
    primitive_func = primitive_instance.get_function()
    array = pd.Series([1, 2, 7, 5])
    answer = pd.Series(primitive_func(array))
    correct_answer = pd.Series(
        [np.nan, 0.7071067811865476, 3.6055512754639887, 1.7311551816712718],
    )
    pd.testing.assert_series_equal(answer, correct_answer)


def test_regular_halflife_std():
    primitive_instance = ExponentialWeightedSTD(halflife=2.7)
    primitive_func = primitive_instance.get_function()
    array = pd.Series([1, 2, 7, 5])
    answer = pd.Series(primitive_func(array))
    correct_answer = pd.Series(
        [np.nan, 0.7071067811865475, 3.3565236098585416, 2.631776826295855],
    )
    pd.testing.assert_series_equal(answer, correct_answer)


def test_regular_alpha_std():
    primitive_instance = ExponentialWeightedSTD(alpha=0.8)
    primitive_func = primitive_instance.get_function()
    array = pd.Series([1, 2, 7, 5])
    answer = pd.Series(primitive_func(array))
    correct_answer = pd.Series(
        [np.nan, 0.7071067811865476, 3.6055512754639887, 1.7311551816712718],
    )
    pd.testing.assert_series_equal(answer, correct_answer)


def test_na_std():
    primitive_instance = ExponentialWeightedSTD(com=0.5)
    primitive_func = primitive_instance.get_function()
    array = pd.Series([1, 2, 7, np.nan, 5])
    answer = pd.Series(primitive_func(array))
    correct_answer = pd.Series(
        [
            np.nan,
            0.7071067811865475,
            3.584153156068229,
            3.5841531560682287,
            1.8408520483016189,
        ],
    )
    pd.testing.assert_series_equal(answer, correct_answer)


def test_ignorena_true_std():
    primitive_instance = ExponentialWeightedSTD(com=0.5, ignore_na=True)
    primitive_func = primitive_instance.get_function()
    array = pd.Series([1, 2, 7, np.nan, 5])
    answer = pd.Series(primitive_func(array))
    correct_answer = pd.Series(
        [
            np.nan,
            0.7071067811865475,
            3.584153156068229,
            3.584153156068229,
            2.0048019276803304,
        ],
    )
    pd.testing.assert_series_equal(answer, correct_answer)


def test_regular_com_var():
    primitive_instance = ExponentialWeightedVariance(com=0.5)
    primitive_func = primitive_instance.get_function()
    array = pd.Series([1, 2, 7, 5])
    answer = pd.Series(primitive_func(array))
    correct_answer = pd.Series(
        [np.nan, 0.49999999999999983, 12.846153846153847, 4.019230769230769],
    )
    pd.testing.assert_series_equal(answer, correct_answer)


def test_regular_span_var():
    primitive_instance = ExponentialWeightedVariance(span=1.5)
    primitive_func = primitive_instance.get_function()
    array = pd.Series([1, 2, 7, 5])
    answer = pd.Series(primitive_func(array))
    correct_answer = pd.Series([np.nan, 0.5, 12.999999999999996, 2.996898263027294])
    pd.testing.assert_series_equal(answer, correct_answer)


def test_regular_halflife_var():
    primitive_instance = ExponentialWeightedVariance(halflife=2.7)
    primitive_func = primitive_instance.get_function()
    array = pd.Series([1, 2, 7, 5])
    answer = pd.Series(primitive_func(array))
    correct_answer = pd.Series(
        [np.nan, 0.49999999999999994, 11.266250743537816, 6.926249263427883],
    )
    pd.testing.assert_series_equal(answer, correct_answer)


def test_regular_alpha_var():
    primitive_instance = ExponentialWeightedVariance(alpha=0.8)
    primitive_func = primitive_instance.get_function()
    array = pd.Series([1, 2, 7, 5])
    answer = pd.Series(primitive_func(array))
    correct_answer = pd.Series([np.nan, 0.5, 12.999999999999996, 2.996898263027294])
    pd.testing.assert_series_equal(answer, correct_answer)


def test_na_var():
    primitive_instance = ExponentialWeightedVariance(com=0.5)
    primitive_func = primitive_instance.get_function()
    array = pd.Series([1, 2, 7, np.nan, 5])
    answer = pd.Series(primitive_func(array))
    correct_answer = pd.Series(
        [
            np.nan,
            0.49999999999999983,
            12.846153846153847,
            12.846153846153843,
            3.3887362637362655,
        ],
    )
    pd.testing.assert_series_equal(answer, correct_answer)


def test_ignorena_true_var():
    primitive_instance = ExponentialWeightedVariance(com=0.5, ignore_na=True)
    primitive_func = primitive_instance.get_function()
    array = pd.Series([1, 2, 7, np.nan, 5])
    answer = pd.Series(primitive_func(array))
    correct_answer = pd.Series(
        [
            np.nan,
            0.49999999999999983,
            12.846153846153847,
            12.846153846153847,
            4.019230769230769,
        ],
    )
    pd.testing.assert_series_equal(answer, correct_answer)
