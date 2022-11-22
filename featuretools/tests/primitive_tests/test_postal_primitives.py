import numpy as np
import pandas as pd
from woodwork.accessor_utils import _is_dask_series, _is_spark_series

from featuretools.primitives.standard.transform.postal import (
    OneDigitPostalCode,
    TwoDigitPostalCode,
)


def generate_actual_and_expected_for_postal_primitive(postal_series, prim):
    prim_ops = {
        "one_digit_postal_code": lambda t: str(t)[0] if pd.notna(t) else np.nan,
        "two_digit_postal_code": lambda t: str(t)[:2] if pd.notna(t) else np.nan,
    }
    op = prim_ops[prim.name]
    actual = prim().get_function()(postal_series)
    if _is_dask_series(postal_series):
        actual = actual.compute()
    if _is_spark_series(postal_series):
        expected = list(map(op, postal_series.to_numpy()))
        actual = [val if pd.notna(val) else np.nan for val in actual.to_numpy()]
    else:
        expected = list(map(op, postal_series))
        actual = [val if pd.notna(val) else np.nan for val in actual.values]
    return actual, expected


def test_one_digit_postal_code(postal_code_dataframe):
    for col in postal_code_dataframe:
        actual, expected = generate_actual_and_expected_for_postal_primitive(
            postal_code_dataframe[col],
            OneDigitPostalCode,
        )
        np.testing.assert_array_equal(actual, expected)


def test_two_digit_postal_code(postal_code_dataframe):
    for col in postal_code_dataframe:
        actual, expected = generate_actual_and_expected_for_postal_primitive(
            postal_code_dataframe[col],
            TwoDigitPostalCode,
        )
        np.testing.assert_array_equal(actual, expected)
