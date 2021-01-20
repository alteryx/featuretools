import pandas as pd
import pytest

import featuretools as ft
from featuretools.variable_types import (
    Categorical,
    Datetime,
    NaturalLanguage,
    Text,
    Timedelta
)


def test_enforces_variable_id_is_str(es):
    assert Categorical("1", es["customers"])

    error_text = 'Variable id must be a string'
    with pytest.raises(AssertionError, match=error_text):
        Categorical(1, es["customers"])


def test_no_column_default_datetime(es):
    variable = Datetime("new_time", es["customers"])
    assert variable.interesting_values.dtype == "datetime64[ns]"

    variable = Timedelta("timedelta", es["customers"])
    assert variable.interesting_values.dtype == "timedelta64[ns]"


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
