import pandas as pd
import pytest

import featuretools as ft
from featuretools.entityset import EntitySet, Relationship
from featuretools.utils.gen_utils import import_or_none

cudf = import_or_none("cudf")


def test_single_table_cudf_entityset_pd_matches_cudf():
    if not cudf:
        pytest.xfail("Fails with cudf not installed")

    primitives_list = [
        "absolute",
        "is_weekend",
        "year",
        "day",
        "num_characters",
        "num_words",
    ]

    pd_es = EntitySet(id="pd_es")
    df = pd.DataFrame(
        {
            "id": [0, 1, 2, 3],
            "values": [1, 12, -34, 27],
            "dates": [
                pd.to_datetime("2019-01-10"),
                pd.to_datetime("2019-02-03"),
                pd.to_datetime("2019-01-01"),
                pd.to_datetime("2017-08-25"),
            ],
            "strings": ["I am a string", "23", "abcdef ghijk", ""],
        },
    )
    ltypes = {"values": "Integer", "dates": "Datetime", "strings": "NaturalLanguage"}
    pd_es.add_dataframe(
        dataframe_name="data",
        dataframe=df,
        index="id",
        logical_types=ltypes,
    )
    pd_fm, _ = ft.dfs(
        entityset=pd_es,
        target_dataframe_name="data",
        trans_primitives=primitives_list,
    )

    cudf_es = ft.EntitySet(id="cudf_es")
    cu_df = cudf.from_pandas(df)
    cudf_es.add_dataframe(
        dataframe_name="data",
        dataframe=cu_df,
        index="id",
        logical_types={"strings": "NaturalLanguage"},
    )
    cudf_fm, _ = ft.dfs(
        entityset=cudf_es,
        target_dataframe_name="data",
        trans_primitives=primitives_list,
    )

    cudf_computed_fm_as_df = cudf_fm.to_pandas()
    cudf_computed_fm_as_df = cudf_computed_fm_as_df.set_index("id")
    pd.testing.assert_frame_equal(pd_fm, cudf_computed_fm_as_df, check_dtype=False)


def test_create_entity_with_non_numeric_index(pd_es, cudf_es):
    df = pd.DataFrame({"id": ["A_1", "A_2", "C", "D"], "values": [1, 12, -34, 27]})
    cudf_df = cudf.from_pandas(df)

    pd_es.add_dataframe(dataframe_name="new_entity", dataframe=df, index="id")

    cudf_es.add_dataframe(
        dataframe_name="new_entity",
        dataframe=cudf_df,
        index="id",
        logical_types={"values": "Integer"},
    )
    pd.testing.assert_frame_equal(
        pd_es["new_entity"].reset_index(drop=True),
        cudf_es["new_entity"].to_pandas(),
        check_dtype=False,
    )
