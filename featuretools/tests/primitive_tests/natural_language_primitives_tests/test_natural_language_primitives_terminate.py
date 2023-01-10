from platform import system

import pandas as pd
import pytest

from featuretools.primitives.utils import _get_natural_language_primitives

TIMEOUT_THRESHOLD = 20

primitives = _get_natural_language_primitives()
print(primitives)


@pytest.mark.timeout(TIMEOUT_THRESHOLD)
@pytest.mark.skipif(
    not (system() == "Linux" or system() == "Darwin"),
    reason="timeout test only supported on UNIX systems",
)
@pytest.mark.parametrize("primitive", primitives)
def test_natlang_primitive_does_not_timeout(
    strings_that_have_triggered_errors_before, primitive
):
    for text in strings_that_have_triggered_errors_before:
        primitive().get_function()(pd.Series(text))
