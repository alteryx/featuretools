import pandas as pd

from featuretools.primitives.standard.transform.postal import (
    OneDigitPostalCode,
    TwoDigitPostalCode,
)


def test_one_digit_postal_code(postal_code_dataframe):
    primitive = OneDigitPostalCode().get_function()
    for x in postal_code_dataframe:
        series = postal_code_dataframe[x]
        actual = primitive(series)
        expected = series.apply(lambda t: str(t)[0] if pd.notna(t) else pd.NA)
        pd.testing.assert_series_equal(actual, expected)


def test_two_digit_postal_code(postal_code_dataframe):
    primitive = TwoDigitPostalCode().get_function()
    for x in postal_code_dataframe:
        series = postal_code_dataframe[x]
        actual = primitive(series)
        expected = series.apply(lambda t: str(t)[:2] if pd.notna(t) else pd.NA)
        pd.testing.assert_series_equal(actual, expected)
