import warnings

import pandas as pd

import featuretools as ft
from featuretools.variable_types import (
    NaturalLanguage,
    Text
)


def test_text_depreciation():
    data = pd.DataFrame({
        "id": [1, 2, 3, 4, 5],
        "text_column": ["a", "c", "b", "a", "a"],
    })

    with warnings.catch_warnings(record=True) as ws:
        warnings.simplefilter("always")
        es = ft.EntitySet()
        es.entity_from_dataframe(entity_id="test", dataframe=data, index="id",
                                 variable_types={"text_column": Text})
    assert len(ws) == 1
    assert ws[0].category == FutureWarning
    assert str(ws[0].message) == "Text has been deprecated. Please use NaturalLanguage instead"

    with warnings.catch_warnings(record=True) as ws:
        warnings.simplefilter("always")
        es = ft.EntitySet()
        es.entity_from_dataframe(entity_id="test", dataframe=data, index="id",
                                 variable_types={"text_column": NaturalLanguage})
    assert len(ws) == 0