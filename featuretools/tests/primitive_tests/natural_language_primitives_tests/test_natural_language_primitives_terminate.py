import signal
from platform import system

import pandas as pd
import pytest

from featuretools.primitives.utils import get_natural_language_primitives

natural_language_primitives = get_natural_language_primitives()
TIMEOUT_THRESHOLD = 20


@pytest.mark.skipif(system() == "Linux" or system() == "Darwin",reason=String)
@pytest.mark.parametrize(
    "primitive",
    natural_language_primitives,
)
def test_natlang_primitive_does_not_timeout(
    strings_that_have_triggered_errors_before, primitive
):
    def handle_SIGALRM(signum, frame):
        raise TimeoutError(
            f"NaturalLanguage primitive took longer than {TIMEOUT_THRESHOLD}"
        )

    signal.signal(signal.SIGALRM, handle_SIGALRM)
    for text in strings_that_have_triggered_errors_before:
        signal.alarm(TIMEOUT_THRESHOLD)
        try:
            primitive().get_function()(pd.Series(text))
        except TimeoutError:
            raise TimeoutError(
                f"NaturalLanguage primitive {primitive.name} took longer than {TIMEOUT_THRESHOLD}"
            )
        else:
            # reset alarm
            signal.alarm(0)
