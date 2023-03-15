from unittest.mock import patch

import pytest
from woodwork.column_schema import ColumnSchema
from woodwork.logical_types import Boolean, Datetime, Double

from featuretools.feature_discovery.feature_discovery import (
    get_features,
    get_matching_columns,
    group_columns,
    index_input_set,
    my_dfs,
)
from featuretools.feature_discovery.type_defs import Feature
from featuretools.primitives import (
    Absolute,
    AddNumeric,
    DateFirstEvent,
    Equal,
    Lag,
    NumUnique,
)
from featuretools.tests.testing_utils.generate_fake_dataframe import (
    generate_fake_dataframe,
)


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
                "Double,numeric": ["f1"],
            },
        ),
    ],
)
@patch.object(Feature, "_generate_hash", lambda x: x.name)
def test_group_columns(column_list, expected):
    column_list = [Feature(*x) for x in column_list]
    actual = group_columns(column_list)
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
