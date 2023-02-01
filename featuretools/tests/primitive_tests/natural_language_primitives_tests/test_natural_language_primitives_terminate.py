import pandas as pd
import pytest

from featuretools.primitives.utils import _get_natural_language_primitives

TIMEOUT_THRESHOLD = 20


class TestNaturalLanguagePrimitivesTerminate:
    # need to sort primitives to avoid pytest collection error
    primitives = sorted(_get_natural_language_primitives().items())

    @pytest.mark.timeout(TIMEOUT_THRESHOLD)
    @pytest.mark.parametrize("primitive", [prim for _, prim in primitives])
    def test_natlang_primitive_does_not_timeout(
        self,
        strings_that_have_triggered_errors_before,
        primitive,
    ):
        for text in strings_that_have_triggered_errors_before:
            primitive().get_function()(pd.Series(text))
