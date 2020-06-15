import databricks.koalas as ks
import numpy as np
import pandas as pd
import pytest

from featuretools.entityset import EntitySet

def test_create_entity_from_koalas_df(pd_es):
    def replace_tuple_columns(pdf):
        new_df = pd.DataFrame()
        for c in pdf.columns:
            if isinstance(pdf[c].iloc[0], tuple):
                new_df[c] = pdf[c].map(lambda x: list(x))
            else:
                new_df[c] = pdf[c]
        return new_df
    
    def replace_nan_with_flag(pdf, flag=-1):
        new_df = pd.DataFrame()
        for c in pdf.columns:
            if isinstance(pdf[c].iloc[0], list):
                new_df[c] = pdf[c].map(lambda l: [flag if np.isnan(x) else x for x in l])
            else:
                new_df[c] = pdf[c]

        return new_df

    cleaned_df = replace_tuple_columns(pd_es["log"].df)
    cleaned_df = replace_nan_with_flag(cleaned_df)
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

def test_create_entity_with_non_numeric_index(pd_es, dask_es):
    df = pd.DataFrame({"id": ["A_1", "A_2", "C", "D"],
                       "values": [1, 12, -34, 27]})
    dask_df = dd.from_pandas(df, npartitions=2)

    pd_es.entity_from_dataframe(
        entity_id="new_entity",
        dataframe=df,
        index="id")

    dask_es.entity_from_dataframe(
        entity_id="new_entity",
        dataframe=dask_df,
        index="id",
        variable_types={"id": ft.variable_types.Id, "values": ft.variable_types.Numeric})

    pd.testing.assert_frame_equal(pd_es['new_entity'].df.reset_index(drop=True), dask_es['new_entity'].df.compute())