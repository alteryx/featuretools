import numpy as np
import pandas as pd
from woodwork.accessor_utils import _is_spark_series

from featuretools.primitives.standard.transform.postal import (
    OneDigitPostalCode,
    TwoDigitPostalCode,
)


def one_digit_postal_code_test(postal_series):
    prim = OneDigitPostalCode().get_function()
    actual = prim(postal_series)
    if _is_spark_series(postal_series):
        expected = [
            str(code)[0] if pd.notna(code) else np.nan
            for code in postal_series.to_numpy()
        ]
        actual = [i if pd.notna(i) else np.nan for i in actual.to_numpy()]
    else:
        expected = [
            str(code)[0] if pd.notna(code) else np.nan for code in postal_series
        ]
        actual = [i if pd.notna(i) else np.nan for i in actual]
    return actual, expected


def two_digit_postal_code_test(postal_series):
    prim = TwoDigitPostalCode().get_function()
    actual = prim(postal_series)
    if _is_spark_series(postal_series):
        expected = [
            str(code)[:2] if pd.notna(code) else np.nan
            for code in postal_series.to_numpy()
        ]
        actual = [i if pd.notna(i) else np.nan for i in actual.to_numpy()]
    else:
        expected = [
            str(code)[:2] if pd.notna(code) else np.nan for code in postal_series
        ]
        actual = [i if pd.notna(i) else np.nan for i in actual.values]
    return actual, expected


def test_one_digit_postal_code(postal_code_dataframes):
    for col in postal_code_dataframes:
        actual, expected = one_digit_postal_code_test(postal_code_dataframes[col])
        np.testing.assert_array_equal(actual, expected)


def test_two_digit_postal_code(postal_code_dataframes):
    for col in postal_code_dataframes:
        actual, expected = two_digit_postal_code_test(postal_code_dataframes[col])
        np.testing.assert_array_equal(actual, expected)
