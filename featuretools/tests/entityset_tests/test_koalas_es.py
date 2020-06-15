import databricks.koalas as ks
import numpy as np
import pandas as pd
import pytest

from featuretools.entityset import EntitySet
from featuretools.tests.testing_utils import pandas_to_koalas_clean

import featuretools as ft

def test_create_entity_from_koalas_df(pd_es):
    cleaned_df = pandas_to_koalas_clean(pd_es["log"].df)
    log_koalas = ks.from_pandas(cleaned_df)

    koalas_es = EntitySet(id="koalas_es")
    koalas_es = koalas_es.entity_from_dataframe(
        entity_id="log_koalas",
        dataframe=log_koalas,
        index="id",
        time_index="datetime",
        variable_types=pd_es["log"].variable_types
    )
    pd.testing.assert_frame_equal(cleaned_df, koalas_es["log_koalas"].df.compute(), check_like=True)

def test_create_entity_with_non_numeric_index(pd_es, koalas_es):
    df = pd.DataFrame({"id": ["A_1", "A_2", "C", "D"],
                       "values": [1, 12, -34, 27]})
    koalas_df = ks.from_pandas(df)

    pd_es.entity_from_dataframe(
        entity_id="new_entity",
        dataframe=df,
        index="id")

    koalas_es.entity_from_dataframe(
        entity_id="new_entity",
        dataframe=koalas_df,
        index="id",
        variable_types={"id": ft.variable_types.Id, "values": ft.variable_types.Numeric})

    pd.testing.assert_frame_equal(pd_es['new_entity'].df.reset_index(drop=True), koalas_df['new_entity'].df.compute())