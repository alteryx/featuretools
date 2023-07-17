import numpy as np
import pandas as pd

from featuretools.primitives import PercentUnique
from featuretools.tests.primitive_tests.utils import (
    PrimitiveTestBase,
)


class TestPercentUnique(PrimitiveTestBase):
    array = pd.Series([1, 1, 2, 2, 3, 4, 5, 6, 7, 8])
    primitive = PercentUnique

    def test_percent_unique(self):
        primitive_func = self.primitive().get_function()
        assert primitive_func(self.array) == (8 / 10.0)

    def test_nans(self):
        primitive_func = self.primitive().get_function()
        array_nans = pd.concat([self.array.copy(), pd.Series([np.nan])])
        assert primitive_func(array_nans) == (8 / 11.0)
        primitive_func = self.primitive(skipna=False).get_function()
        assert primitive_func(array_nans) == (9 / 11.0)

    def test_multiple_nans(self):
        primitive_func = self.primitive().get_function()
        array_nans = pd.concat([self.array.copy(), pd.Series([np.nan] * 3)])
        assert primitive_func(array_nans) == (8 / 13.0)
        primitive_func = self.primitive(skipna=False).get_function()
        assert primitive_func(array_nans) == (9 / 13.0)

    def test_empty_string(self):
        primitive_func = self.primitive().get_function()
        array_empty_string = pd.concat([self.array.copy(), pd.Series([np.nan, "", ""])])
        assert primitive_func(array_empty_string) == (9 / 13.0)
