import pandas as pd

from featuretools.primitives.standard.transform.postal import (
    OneDigitPostalCode,
    TwoDigitPostalCode,
)


def test_single_digit_postal_code(postal_code_series_pd):
    prim = OneDigitPostalCode().get_function()
    expected = pd.Series([code[0] for code in postal_code_series_pd])
    actual = prim(postal_code_series_pd)
    pd.testing.assert_series_equal(expected, actual)


def test_two_digit_postal_code(postal_code_series_pd):
    prim = TwoDigitPostalCode().get_function()
    expected = pd.Series([code[:1] for code in postal_code_series_pd])
    actual = prim(postal_code_series_pd)
    pd.testing.assert_series_equal(expected, actual)
