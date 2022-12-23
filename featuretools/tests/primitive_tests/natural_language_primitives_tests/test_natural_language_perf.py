import signal
from time import sleep

import pandas as pd
import pytest

from featuretools.primitives import (
    CountString,
    MeanCharactersPerWord,
    MedianWordLength,
    NumberOfHashtags,
    NumberOfMentions,
    NumberOfUniqueWords,
    NumberOfWordsInQuotes,
    PunctuationCount,
    UpperCaseCount,
    WhitespaceCount,
)

TIMEOUT_THRESHOLD = 20


@pytest.mark.parametrize(
    "primitive",
    [
        CountString,
        MeanCharactersPerWord,
        MedianWordLength,
        NumberOfHashtags,
        NumberOfMentions,
        NumberOfUniqueWords,
        NumberOfWordsInQuotes,
        PunctuationCount,
        UpperCaseCount,
        WhitespaceCount,
    ],
)
def test_natlang_primitive_does_not_timeout(shuffled_test_strings, primitive):
    def handle_SIGALRM(signum, frame):
        raise TimeoutError(
            f"NaturalLanguage primitive took longer than {TIMEOUT_THRESHOLD}"
        )

    signal.signal(signal.SIGALRM, handle_SIGALRM)
    for text in shuffled_test_strings:
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
