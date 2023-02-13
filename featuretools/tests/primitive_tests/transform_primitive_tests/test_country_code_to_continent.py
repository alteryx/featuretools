import numpy as np
import pandas as pd

from featuretools.primitives import CountryCodeToContinent
from featuretools.tests.primitive_tests.utils import (
    PrimitiveTestBase,
    find_applicable_primitives,
    valid_dfs,
)


class TestCountryCodeToContinent(PrimitiveTestBase):
    primitive = CountryCodeToContinent

    def test_country_codes(self):
        primitive_instance = CountryCodeToContinent
        primitive_func = primitive_instance().get_function()
        array = pd.Series(["AM", "SOM", 780])
        answer = pd.Series(["Asia", "Africa", "North America"])
        pd.testing.assert_series_equal(
            primitive_func(array),
            answer,
            check_names=False,
        )

    def test_invalid_codes(self):
        primitive_instance = CountryCodeToContinent
        primitive_func = primitive_instance().get_function()
        array = pd.Series(["AM", "FZ", "ZZZ", 999])
        answer = pd.Series(["Asia", np.nan, np.nan, np.nan])
        pd.testing.assert_series_equal(
            primitive_func(array),
            answer,
            check_names=False,
        )

    def test_missing_codes(self):
        primitive_instance = CountryCodeToContinent
        primitive_func = primitive_instance().get_function()
        array = pd.Series(["AM", np.nan, None, ""])
        answer = pd.Series(["Asia", np.nan, np.nan, np.nan])
        pd.testing.assert_series_equal(
            primitive_func(array),
            answer,
            check_names=False,
        )

    def test_with_featuretools(self, es):
        transform, aggregation = find_applicable_primitives(self.primitive)
        primitive_instance = self.primitive()
        transform.append(primitive_instance)
        valid_dfs(es, aggregation, transform, self.primitive.name.upper())
