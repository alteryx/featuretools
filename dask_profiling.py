import copy

import pandas as pd
from dask import dataframe as dd

import featuretools as ft
from featuretools.tests.testing_utils import make_ecommerce_entityset


def make_dask_es(es):
    dask_es = copy.deepcopy(es)
    for entity in dask_es.entities:
        entity.df = dd.from_pandas(entity.df.reset_index(drop=True), npartitions=2)
    return dask_es

@profile
def test_transform(es, dask_es):
    primitives = ft.list_primitives()
    trans_list = primitives[primitives['type'] == 'transform']['name'].tolist()
    # These primitives currently do not work
    bad_primitives = ['cum_mean', 'time_since', 'equal', 'not_equal', 'equal_scalar', 'not_equal_scalar']
    trans_primitives = [prim for prim in trans_list if prim not in bad_primitives]
    agg_primitives = []

    # Run DFS using each entity as a target and confirm results match
    for entity in es.entities:
        fm, _ = ft.dfs(entityset=es,
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

@profile
def test_aggregation(es, dask_es):
    primitives = ft.list_primitives()
    trans_primitives = []
    agg_list = primitives[primitives['type'] == 'aggregation']['name'].tolist()
    bad_primitives = ['trend', 'first', 'last', 'time_since_first', 'n_most_common', 'time_since_last']
    agg_primitives = [prim for prim in agg_list if prim not in bad_primitives]

    # Run DFS using each entity as a target and confirm results match
    for entity in es.entities:
        fm, _ = ft.dfs(entityset=es,
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


if __name__ == "__main__":
    es = make_ecommerce_entityset()
    dask_es = make_dask_es(es)

    test_aggregation(es, dask_es)
    test_transform(es, dask_es)
