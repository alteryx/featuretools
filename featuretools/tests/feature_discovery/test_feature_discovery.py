from unittest.mock import patch

import pytest
from woodwork.column_schema import ColumnSchema
from woodwork.logical_types import Boolean, Double

from featuretools.feature_discovery.feature_discovery import (
    get_features,
    get_matching_columns,
    group_columns,
    index_input_set,
)
from featuretools.feature_discovery.type_defs import Feature
from featuretools.primitives import AddNumeric


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
            [[["f1", "f2"], ["f1", "f3"], ["f2", "f3"]]],
        ),
    ],
)
def test_get_matching_columns(col_groups, primitive, expected):
    actual = get_matching_columns(col_groups, primitive)

    assert actual == expected
