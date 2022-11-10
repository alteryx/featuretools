import numpy as np
import pandas as pd

from featuretools.primitives.standard.transform.postal import (
    OneDigitPostalCode,
    TwoDigitPostalCode,
)


def test_single_digit_postal_code(postal_code_series_pd):
    print(postal_code_series_pd)
    prim = OneDigitPostalCode().get_function()
    actual = prim(postal_code_series_pd)
    expected = [
        str(code)[0] if not pd.isna(code) else code for code in postal_code_series_pd
    ]
    pd.testing.assert_series_equal(pd.Series(actual), pd.Series(expected))


def test_two_digit_postal_code(postal_code_series_pd):
    prim = TwoDigitPostalCode().get_function()
    actual = prim(postal_code_series_pd)
    expected = [
        str(code)[:2] if not pd.isna(code) else code for code in postal_code_series_pd
    ]
    pd.testing.assert_series_equal(pd.Series(actual), pd.Series(expected))
