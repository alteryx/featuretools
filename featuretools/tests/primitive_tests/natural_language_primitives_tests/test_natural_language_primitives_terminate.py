import pandas as pd
import pytest

from featuretools.primitives.utils import _get_natural_language_primitives

TIMEOUT_THRESHOLD = 20


class TestNaturalLanguagePrimitivesTerminate:

    primitives = _get_natural_language_primitives()

    @pytest.mark.parametrize("primitive", primitives)
    def test_natlang_primitive_does_not_timeout(
        self,
        strings_that_have_triggered_errors_before,
        primitive,
    ):
        print(f"Primitive {primitive}")
        for text in strings_that_have_triggered_errors_before:
            primitive().get_function()(pd.Series(text))
