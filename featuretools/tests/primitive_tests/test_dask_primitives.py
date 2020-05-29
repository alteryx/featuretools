import pandas as pd

import featuretools as ft
from featuretools.primitives import (
    get_aggregation_primitives,
    get_transform_primitives
)

UNSUPPORTED = [p.name for p in get_transform_primitives().values() if not p.dask_compatible]
UNSUPPORTED += [p.name for p in get_aggregation_primitives().values() if not p.dask_compatible]


def test_transform(pd_es, dask_es):
    primitives = ft.list_primitives()
    trans_list = primitives[primitives['type'] == 'transform']['name'].tolist()
    trans_primitives = [prim for prim in trans_list if prim not in UNSUPPORTED]
    agg_primitives = []

    assert pd_es == dask_es
    # Run DFS using each entity as a target and confirm results match
    for entity in pd_es.entities:
        fm, _ = ft.dfs(entityset=pd_es,
                       target_entity=entity.id,
                       trans_primitives=trans_primitives,
                       agg_primitives=agg_primitives,
                       cutoff_time=pd.Timestamp("2019-01-05 04:00"),
                       max_depth=2,
                       max_features=100)

        dask_fm, _ = ft.dfs(entityset=dask_es,
                            target_entity=entity.id,
                            trans_primitives=trans_primitives,
                            agg_primitives=agg_primitives,
                            cutoff_time=pd.Timestamp("2019-01-05 04:00"),
                            max_depth=2,
                            max_features=100)
        # Use the same columns and make sure both indexes are sorted the same
        dask_computed_fm = dask_fm.compute().set_index(entity.index).loc[fm.index][fm.columns]
        pd.testing.assert_frame_equal(fm, dask_computed_fm)


def test_aggregation(pd_es, dask_es):
    primitives = ft.list_primitives()
    trans_primitives = []
    agg_list = primitives[primitives['type'] == 'aggregation']['name'].tolist()
    agg_primitives = [prim for prim in agg_list if prim not in UNSUPPORTED]

    assert pd_es == dask_es

    # Run DFS using each entity as a target and confirm results match
    for entity in pd_es.entities:
        fm, _ = ft.dfs(entityset=pd_es,
                       target_entity=entity.id,
                       trans_primitives=trans_primitives,
                       agg_primitives=agg_primitives,
                       cutoff_time=pd.Timestamp("2019-01-05 04:00"),
                       max_depth=2)

        dask_fm, _ = ft.dfs(entityset=dask_es,
                            target_entity=entity.id,
                            trans_primitives=trans_primitives,
                            agg_primitives=agg_primitives,
                            cutoff_time=pd.Timestamp("2019-01-05 04:00"),
                            max_depth=2)
        # Use the same columns and make sure both indexes are sorted the same
        dask_computed_fm = dask_fm.compute().set_index(entity.index).loc[fm.index][fm.columns]
        pd.testing.assert_frame_equal(fm, dask_computed_fm, check_dtype=False)
