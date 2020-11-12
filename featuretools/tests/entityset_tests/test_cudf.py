import featuretools as ft
import pandas as pd
import dask.dataframe as dd
import cudf


def test_cudf_basic():
    id = [0, 1, 2, 3, 4]
    values = [12, -35, 14, 103, -51]


    df = cudf.DataFrame({"id": id, "values": values})
    es = ft.EntitySet(id="cudf_es")

    es = es.entity_from_dataframe(entity_id="cudf_entity",
                                dataframe=df,
                                index="id",
                                variable_types={"id": ft.variable_types.Id,
                                                "values": ft.variable_types.Numeric})


    feature_matrix, features = ft.dfs(entityset=es,
                                    target_entity="cudf_entity",
                                    trans_primitives=["negate"])



test_cudf_basic()