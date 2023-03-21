from typing import List, cast
from unittest.mock import patch

import pytest
from woodwork.column_schema import ColumnSchema
from woodwork.logical_types import Boolean, Datetime, Double, Ordinal

from featuretools.entityset.entityset import EntitySet
from featuretools.feature_discovery.feature_discovery import (
    features_from_primitive,
    get_features,
    get_matching_columns,
    group_features,
    index_input_set,
    my_dfs,
)
from featuretools.feature_discovery.type_defs import Feature
from featuretools.primitives import (
    Absolute,
    AddNumeric,
    DateFirstEvent,
    Equal,
    ExpandingCount,
    ExpandingMax,
    ExpandingMean,
    ExpandingMin,
    ExpandingSTD,
    ExpandingTrend,
    Lag,
    NumUnique,
    RollingCount,
    RollingMax,
    RollingMean,
    RollingMin,
    RollingOutlierCount,
    RollingSTD,
    RollingTrend,
    SubtractNumeric,
)
from featuretools.primitives.utils import get_transform_primitives
from featuretools.synthesis import dfs
from featuretools.tests.testing_utils.generate_fake_dataframe import (
    generate_fake_dataframe,
)

DEFAULT_LT_FOR_TAG = {
    "category": Ordinal,
    "numeric": Double,
    "time_index": Datetime,
}


ROLLING_PRIMITIVES = [
    RollingMax,
    RollingMin,
    RollingMean,
    RollingSTD,
    RollingTrend,
    RollingCount,
    RollingOutlierCount,
]

EXPANDING_PRIMITIVES = [
    ExpandingMax,
    ExpandingMin,
    ExpandingMean,
    ExpandingSTD,
    ExpandingTrend,
    ExpandingCount,
]

TIME_SERIES_PRIMITIVES = ROLLING_PRIMITIVES + EXPANDING_PRIMITIVES + [Lag]

ROLLING_PRIMITIVES_BY_NAME = {p.name: p for p in ROLLING_PRIMITIVES}
EXPANDING_PRIMITIVES_BY_NAME = {p.name: p for p in EXPANDING_PRIMITIVES}
TIME_SERIES_PRIMITIVES_BY_NAME = {p.name: p for p in TIME_SERIES_PRIMITIVES}

PRIMITIVES_RELYING_ON_ORDERING = [
    "absolute_diff",
    "cum_sum",
    "cum_count",
    "cum_mean",
    "cum_max",
    "cumulative_time_since_last_false",  # time_index
    "cumulative_time_since_last_true",  # time_index
    "diff",
    "diff_datetime",
    "exponential_weighted_average",
    "exponential_weighted_std",
    "exponential_weighted_variance",
    "greater_than_previous",
    "is_first_occurrence",
    "is_last_occurrence",
    "is_max_so_far",
    "is_min_so_far",
    "lag",  # time_index
    "less_than_previous",
    "percent_change",
    "same_as_previous",
    "time_since_previous",  # time_index
]

PRIMITIVES_WITH_ISSUES = [
    # // as of premium - primitives 0.14.0 and feature - tools 0.26.1
    "multiply_boolean",  # functionality duplicated by 'and' primitive
    "numeric_lag",  # deperecated and replaced with `lag`
]

PRIMITIVES_REQUIRING_INPUT = [
    # // as of premium - primitives 0.14.0 and feature - tools 0.26.1
    "count_string",
    "distance_to_holiday",
    "is_in_geobox",
    "score_percentile",
    "subtract_numeric_scalar",
    "scalar_subtract_numeric_feature",
    "not_equal_scalar",
    "multiply_numeric_scalar",
    "modulo_numeric_scalar",
    "divide_numeric_scalar",
    "add_numeric_scalar",
    "equal_scalar",
    "greater_than_equal_to_scalar",
    "less_than_equal_to_scalar",
    "divide_by_feature",
    "greater_than_scalar",
    "less_than_scalar",
    "modulo_by_feature",
    "time_since",
    "savgol_filter",
    "numeric_lag",  # duplicate of lag
    "isin",
    "numeric_bin",
]

PRIMITIVE_BLACKLIST = (
    PRIMITIVES_REQUIRING_INPUT
    + PRIMITIVES_WITH_ISSUES
    + PRIMITIVES_RELYING_ON_ORDERING
    + [x for x in TIME_SERIES_PRIMITIVES_BY_NAME]
)


def get_valid_tempo_transform_primitives():
    all_available_trans_prims = []
    for prim_name, prim_cls in get_transform_primitives().items():
        if prim_name not in PRIMITIVE_BLACKLIST:
            all_available_trans_prims.append(prim_cls)

    return all_available_trans_prims


@pytest.mark.parametrize(
    "column_list, expected",
    [
        ([ColumnSchema(logical_type=Boolean)], {"Boolean": 1}),
        ([ColumnSchema()], {"ANY": 1}),
        (
            [
                ColumnSchema(logical_type=Boolean),
                ColumnSchema(logical_type=Boolean),
            ],
            {"Boolean": 2},
        ),
    ],
)
def test_index_input_set(column_list, expected):
    actual = index_input_set(column_list)

    assert actual == expected


@pytest.mark.parametrize(
    "column_list, expected",
    [
        (
            [("f1", Boolean), ("f2", Boolean), ("f3", Boolean)],
            {"ANY": ["f1", "f2", "f3"], "Boolean": ["f1", "f2", "f3"]},
        ),
        (
            [("f1", Double), ("f2", Double), ("f3", Double)],
            {
                "ANY": ["f1", "f2", "f3"],
                "Double": ["f1", "f2", "f3"],
                "numeric": ["f1", "f2", "f3"],
                "Double,numeric": ["f1", "f2", "f3"],
            },
        ),
        (
            [("f1", Datetime, {"time_index"}), ("f2", Double)],
            {
                "ANY": ["f1", "f2"],
                "Datetime": ["f1"],
                "time_index": ["f1"],
                "Datetime,time_index": ["f1"],
                "Double": ["f2"],
                "numeric": ["f2"],
                "Double,numeric": ["f2"],
            },
        ),
    ],
)
@patch.object(Feature, "_generate_hash", lambda x: x.name)
def test_group_features(column_list, expected):
    column_list = [Feature(*x) for x in column_list]
    actual = group_features(column_list)
    actual = {k: [x.id for x in v] for k, v in actual.items()}
    assert actual == expected


@pytest.mark.parametrize(
    "col_groups, input_set, commutative, expected",
    [
        (
            {"ANY": ["f1", "f2", "f3"], "Boolean": ["f1", "f2", "f3"]},
            [ColumnSchema(logical_type=Boolean)],
            False,
            [["f1"], ["f2"], ["f3"]],
        ),
        (
            {"ANY": ["f1", "f2"], "Boolean": ["f1", "f2"]},
            [ColumnSchema(logical_type=Boolean), ColumnSchema(logical_type=Boolean)],
            False,
            [["f1", "f2"], ["f2", "f1"]],
        ),
        (
            {"ANY": ["f1", "f2"], "Boolean": ["f1", "f2"]},
            [ColumnSchema(logical_type=Boolean), ColumnSchema(logical_type=Boolean)],
            True,
            [["f1", "f2"]],
        ),
        (
            {
                "ANY": ["f1"],
                "Datetime": ["f1"],
                "time_index": ["f1"],
                "Datetime,time_index": ["f1"],
            },
            [ColumnSchema(logical_type=Datetime, semantic_tags={"time_index"})],
            False,
            [["f1"]],
        ),
    ],
)
def test_get_features(col_groups, input_set, commutative, expected):
    actual = get_features(col_groups, input_set, commutative)
    assert actual == expected


@pytest.mark.parametrize(
    "col_groups, primitive, expected",
    [
        (
            {
                "ANY": ["f1", "f2", "f3"],
                "Double": ["f1", "f2", "f3"],
                "numeric": ["f1", "f2", "f3"],
            },
            AddNumeric,
            [["f1", "f2"], ["f1", "f3"], ["f2", "f3"]],
        ),
        (
            {
                "ANY": ["f1", "f2", "f3"],
                "Boolean": ["f1", "f2", "f3"],
            },
            AddNumeric,
            [],
        ),
        (
            {
                "ANY": ["f1"],
                "Datetime": ["f1"],
            },
            DateFirstEvent,
            [],
        ),
        (
            {
                "ANY": ["f1"],
                "time_index": ["f1"],
            },
            DateFirstEvent,
            [],
        ),
        (
            {
                "ANY": ["f1"],
                "Datetime": ["f1"],
                "time_index": ["f1"],
                "Datetime,time_index": ["f1"],
            },
            DateFirstEvent,
            [["f1"]],
        ),
        (
            {
                "ANY": ["f1", "f2"],
                "Datetime": ["f1"],
                "time_index": ["f1"],
                "Datetime,time_index": ["f1"],
                "Double": ["f2"],
                "numeric": ["f2"],
                "Double,numeric": ["f2"],
            },
            NumUnique,
            [],
        ),
        (
            {
                "ANY": ["f1", "f2", "f3"],
                "Datetime": ["f1"],
                "time_index": ["f1"],
                "Datetime,time_index": ["f1"],
                "Double": ["f2"],
                "numeric": ["f2"],
                "Double,numeric": ["f2"],
                "Ordinal": ["f3"],
                "category": ["f3"],
                "Ordinal,category": ["f3"],
            },
            NumUnique,
            [["f3"]],
        ),
        (
            {
                "ANY": ["f1", "f2", "f3"],
                "Datetime": ["f1"],
                "time_index": ["f1"],
                "Datetime,time_index": ["f1"],
                "Double": ["f2"],
                "numeric": ["f2"],
                "Double,numeric": ["f2"],
                "Ordinal": ["f3"],
                "category": ["f3"],
                "Ordinal,category": ["f3"],
            },
            Equal,
            [["f1", "f2"], ["f1", "f3"], ["f2", "f3"]],
        ),
        (
            {
                "ANY": ["f1", "f2", "f3", "f4", "f5"],
                "Datetime": ["f1"],
                "time_index": ["f1"],
                "Datetime,time_index": ["f1"],
                "Ordinal": ["f2"],
                "category": ["f2"],
                "Ordinal,category": ["f2"],
                "Double": ["f3"],
                "numeric": ["f3"],
                "Double,numeric": ["f3"],
                "Boolean": ["f4"],
                "BooleanNullable": ["f5"],
            },
            Lag,
            [["f2", "f1"], ["f3", "f1"], ["f4", "f1"], ["f5", "f1"]],
        ),
    ],
)
def test_get_matching_columns(col_groups, primitive, expected):
    actual = get_matching_columns(col_groups, primitive)

    assert actual == expected


@pytest.mark.parametrize(
    "col_defs, primitives, expected",
    [
        (
            [
                ("f_1", "Double"),
                ("f_2", "Double"),
                ("f_3", "Boolean"),
                ("f_4", "Double"),
            ],
            [AddNumeric],
            {"f_1 + f_2", "f_1 + f_4", "f_2 + f_4"},
        ),
        (
            [
                ("f_1", "Double"),
                ("f_2", "Double"),
            ],
            [Absolute],
            {"ABSOLUTE(f_1)", "ABSOLUTE(f_2)"},
        ),
    ],
)
def test_new_dfs(col_defs, primitives, expected):

    input_feature_names = set([x[0] for x in col_defs])
    df = generate_fake_dataframe(
        col_defs=col_defs,
    )

    all_features = my_dfs(df.ww.schema, primitives)

    new_feature_names = set([x.name for x in all_features]) - input_feature_names
    assert new_feature_names == expected


def get_default_logical_type(tags: set[str]):
    for tag, lt in DEFAULT_LT_FOR_TAG.items():
        if tag in tags:
            return lt

    raise Exception(f"NO DEFAULT LOGICAL TYPE FOR TAGS: {tags}")


@pytest.mark.parametrize(
    "primitive",
    get_valid_tempo_transform_primitives(),
)
def test_features_from_primitive(primitive):
    # primitive = Absolute
    input_list = primitive.input_types
    if not isinstance(input_list[0], list):
        input_list = [input_list]

    input_list = cast(List[List[ColumnSchema]], input_list)
    assert isinstance(input_list, List)
    assert isinstance(input_list[0], List)
    assert isinstance(input_list[0][0], ColumnSchema)

    test_features = []
    for input_set in input_list:
        for idx, col_schema in enumerate(input_set):
            logical_type = col_schema.logical_type
            semantic_tags = col_schema.semantic_tags
            if logical_type is not None:
                logical_type = type(logical_type)
            elif len(semantic_tags) > 0:
                logical_type = get_default_logical_type(semantic_tags)
            else:
                logical_type = Double

            test_features.append(
                Feature(f"f_{idx}", logical_type, semantic_tags),
            )

    col_groups = group_features(test_features)
    generated_features = features_from_primitive(primitive, col_groups)

    assert len(generated_features) > 0


@pytest.mark.parametrize(
    "col_defs, primitives",
    [
        (
            [
                ("f_1", "Double"),
                ("f_2", "Double"),
                ("f_3", "Boolean"),
                ("f_4", "Boolean"),
                ("f_5", "Double"),
            ],
            [AddNumeric, Absolute, SubtractNumeric],
        ),
    ],
)
def test_compare_dfs(col_defs, primitives):
    input_feature_names = set([x[0] for x in col_defs])
    df = generate_fake_dataframe(
        col_defs=col_defs,
        include_index=True,
    )

    all_features = my_dfs(df.ww.schema, primitives)

    es = EntitySet(id="nums")
    es.add_dataframe(df, "nums", index="idx")

    fdefs = dfs(
        entityset=es,
        target_dataframe_name="nums",
        trans_primitives=primitives,
        features_only=True,
    )

    new_feature_names1 = set([x.name for x in all_features]) - input_feature_names

    new_feature_names2 = set([x.get_name() for x in fdefs]) - input_feature_names

    assert new_feature_names1 == new_feature_names2
