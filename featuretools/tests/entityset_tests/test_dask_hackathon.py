import os

import pandas as pd
from dask import dataframe as dd

import featuretools as ft
from featuretools.entityset import EntitySet


def test_hackathon_single_table():
    data_path = os.path.join(os.path.dirname(__file__), "data/hackathon_users_data.csv")
    df = pd.read_csv(data_path)
    es = EntitySet(id='es')
    es.entity_from_dataframe(
        entity_id="users",
        dataframe=df,
        index="RESPID",
    )

    trans_primitives = ['cum_sum', 'diff', 'absolute', 'is_weekend', 'year', 'day', 'num_characters', 'num_words']

    fm, _ = ft.dfs(entityset=es,
                   target_entity="users",
                   trans_primitives=trans_primitives)

    df_dd = dd.read_csv(data_path, blocksize='1MB')
    dask_es = EntitySet(id="dask_es")
    dask_es.entity_from_dataframe(
        entity_id="users",
        dataframe=df_dd,
        index="RESPID",
    )
    dask_fm, _ = ft.dfs(entityset=dask_es,
                        target_entity="users",
                        trans_primitives=trans_primitives)

    assert es == dask_es
    # Account for difference in index and column ordering when making comarisons
    assert es['users'].df.reset_index(drop=True).equals(dask_es['users'].df.compute().reset_index(drop=True))
    # Use the same columns and make sure both are sorted on index values
    dask_computed_fm = dask_fm.compute().set_index("RESPID").loc[fm.index][fm.columns]
    pd.testing.assert_frame_equal(fm, dask_computed_fm)
