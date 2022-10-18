import pytest
from woodwork.column_schema import ColumnSchema

from featuretools.primitives import (
    AggregationPrimitive,
    Count,
    Hour,
    IsIn,
    Not,
    TimeSincePrevious,
    TransformPrimitive,
)
from featuretools.synthesis.get_valid_primitives import get_valid_primitives
from featuretools.utils.gen_utils import Library


def test_get_valid_primitives_selected_primitives(es):
    agg_prims, trans_prims = get_valid_primitives(
        es,
        "log",
        selected_primitives=[Hour, Count],
    )
    assert set(agg_prims) == set([Count])
    assert set(trans_prims) == set([Hour])

    agg_prims, trans_prims = get_valid_primitives(
        es,
        "products",
        selected_primitives=[Hour],
        max_depth=1,
    )
    assert set(agg_prims) == set()
    assert set(trans_prims) == set()


def test_get_valid_primitives_selected_primitives_strings(es):
    agg_prims, trans_prims = get_valid_primitives(
        es,
        "log",
        selected_primitives=["hour", "count"],
    )
    assert set(agg_prims) == set([Count])
    assert set(trans_prims) == set([Hour])

    agg_prims, trans_prims = get_valid_primitives(
        es,
        "products",
        selected_primitives=["hour"],
        max_depth=1,
    )
    assert set(agg_prims) == set()
    assert set(trans_prims) == set()


def test_invalid_primitive(es):
    with pytest.raises(ValueError, match="'foobar' is not a recognized primitive name"):
        get_valid_primitives(
            es,
            target_dataframe_name="log",
            selected_primitives=["foobar"],
        )

    msg = (
        "Selected primitive <enum 'Library'> "
        "is not an AggregationPrimitive, TransformPrimitive, or str"
    )
    with pytest.raises(ValueError, match=msg):
        get_valid_primitives(
            es,
            target_dataframe_name="log",
            selected_primitives=[Library],
        )


def test_primitive_compatibility(es):
    _, trans_prims = get_valid_primitives(
        es,
        "customers",
        selected_primitives=[TimeSincePrevious],
    )

    if es.dataframe_type != Library.PANDAS:
        assert len(trans_prims) == 0
    else:
        assert len(trans_prims) == 1


def test_get_valid_primitives_custom_primitives(pd_es):
    class ThreeMostCommonCat(AggregationPrimitive):
        name = "n_most_common_categorical"
        input_types = [ColumnSchema(semantic_tags={"category"})]
        return_type = ColumnSchema(semantic_tags={"category"})
        number_output_features = 3

    class AddThree(TransformPrimitive):
        name = "add_three"
        input_types = [
            ColumnSchema(semantic_tags="numeric"),
            ColumnSchema(semantic_tags="numeric"),
            ColumnSchema(semantic_tags="numeric"),
        ]
        return_type = ColumnSchema(semantic_tags="numeric")
        commutative = True
        compatibility = [Library.PANDAS, Library.DASK, Library.SPARK]

    agg_prims, trans_prims = get_valid_primitives(pd_es, "log")
    assert ThreeMostCommonCat not in agg_prims
    assert AddThree not in trans_prims

    with pytest.raises(
        ValueError,
        match="'add_three' is not a recognized primitive name",
    ):
        agg_prims, trans_prims = get_valid_primitives(
            pd_es,
            "log",
            2,
            [ThreeMostCommonCat, "add_three"],
        )


def test_get_valid_primitives_all_primitives(es):
    agg_prims, trans_prims = get_valid_primitives(es, "customers")
    assert Count in agg_prims
    assert Hour in trans_prims


def test_get_valid_primitives_single_table(transform_es):
    msg = "Only one dataframe in entityset, changing max_depth to 1 since deeper features cannot be created"
    with pytest.warns(UserWarning, match=msg):
        agg_prims, trans_prims = get_valid_primitives(transform_es, "first")

    assert set(agg_prims) == set()
    assert IsIn in trans_prims


def test_get_valid_primitives_with_dfs_kwargs(es):
    agg_prims, trans_prims = get_valid_primitives(
        es,
        "customers",
        selected_primitives=[Hour, Count, Not],
    )
    assert set(agg_prims) == set([Count])
    assert set(trans_prims) == set([Hour, Not])

    # Can use other dfs parameters and they get applied
    agg_prims, trans_prims = get_valid_primitives(
        es,
        "customers",
        selected_primitives=[Hour, Count, Not],
        ignore_columns={"customers": ["loves_ice_cream"]},
    )
    assert set(agg_prims) == set([Count])
    assert set(trans_prims) == set([Hour])

    agg_prims, trans_prims = get_valid_primitives(
        es,
        "products",
        selected_primitives=[Hour, Count],
        ignore_dataframes=["log"],
    )
    assert set(agg_prims) == set()
    assert set(trans_prims) == set()
