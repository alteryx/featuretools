import pandas as pd
import pytest

import featuretools as ft
from featuretools.variable_types import NaturalLanguage, Text


def test_text_depreciation():
    data = pd.DataFrame({
        "id": [1, 2, 3, 4, 5],
        "text_column": ["a", "c", "b", "a", "a"],
    })
    es = ft.EntitySet()
    match = "Text has been deprecated. Please use NaturalLanguage instead"
    with pytest.warns(FutureWarning, match=match):
        es.entity_from_dataframe(entity_id="test", dataframe=data, index="id",
                                 variable_types={"text_column": Text})
    es = ft.EntitySet()
    with pytest.warns(None) as record:
        es.entity_from_dataframe(entity_id="test", dataframe=data, index="id",
                                 variable_types={"text_column": NaturalLanguage})
        assert len(record) == 0
