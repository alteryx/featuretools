from typing import Set
from unittest.mock import patch

import pytest
from woodwork.column_schema import ColumnSchema
from woodwork.logical_types import Boolean, Datetime, Double, Ordinal

from featuretools.entityset.entityset import EntitySet
from featuretools.feature_discovery.feature_discovery import (
    feature_to_keys,
    generate_hashing_keys_from_column_schema,
    get_features,
    get_matching_features,
    group_features,
    index_column_set,
    my_dfs,
    schema_to_features,
)
from featuretools.feature_discovery.type_defs import Feature
from featuretools.primitives import (
    Absolute,
    AddNumeric,
    DateFirstEvent,
    Day,
    Equal,
    Lag,
    NumUnique,
    SubtractNumeric,
)
from featuretools.primitives.standard.transform.binary.multiply_numeric_boolean import (
    MultiplyNumericBoolean,
)
from featuretools.synthesis import dfs
from featuretools.tests.testing_utils.generate_fake_dataframe import (
    generate_fake_dataframe,
)

DEFAULT_LT_FOR_TAG = {
    "category": Ordinal,
    "numeric": Double,
    "time_index": Datetime,
}


@pytest.mark.parametrize(
    "column_schema, expected",
    [
        (ColumnSchema(logical_type=Double), ["Double"]),
        (ColumnSchema(semantic_tags={"index"}), ["index"]),
        (
            ColumnSchema(logical_type=Double, semantic_tags={"index", "other"}),
            ["Double,index", "Double,other"],
        ),
    ],
)
def test_column_to_keys(column_schema, expected):
    actual = generate_hashing_keys_from_column_schema(column_schema)
    assert set(actual) == set(expected)


@pytest.mark.parametrize(
    "feature, expected",
    [
        (("f1", Double), ["Double", "numeric", "Double,numeric", "ANY"]),
        (
            ("f1", Datetime, {"time_index"}),
            ["Datetime", "time_index", "Datetime,time_index", "ANY"],
        ),
        (
            ("f1", Double, {"index", "other"}),
            ["Double", "index", "other", "Double,index", "Double,other", "ANY"],
        ),
    ],
)
def test_feature_to_keys(feature, expected):
    actual = feature_to_keys(Feature(*feature))
    assert set(actual) == set(expected)


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
    actual = index_column_set(column_list)

    assert actual == expected


@pytest.mark.parametrize(
    "column_list, expected",
    [
        (
            [("f1", Boolean), ("f2", Boolean), ("f3", Boolean)],
            {"ANY": ["f1", "f2", "f3"], "Boolean": ["f1", "f2", "f3"]},
        ),
        (
            [("f1", Double), ("f2", Double), ("f3", Double, {"index"})],
            {
                "ANY": ["f1", "f2", "f3"],
                "Double": ["f1", "f2", "f3"],
                "numeric": ["f1", "f2"],
                "Double,numeric": ["f1", "f2"],
                "index": ["f3"],
                "Double,index": ["f3"],
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
    "feature_groups, primitive, expected",
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
                "ANY": ["f1", "f2"],
                "Double": ["f1"],
                "numeric": ["f1"],
                "Double,numeric": ["f1"],
                "Boolean": ["f2"],
            },
            MultiplyNumericBoolean,
            [["f1", "f2"]],
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
def test_get_matching_features(feature_groups, primitive, expected):
    actual = get_matching_features(feature_groups, primitive)

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
@patch.object(Feature, "_generate_hash", lambda x: x.name)
def test_new_dfs(col_defs, primitives, expected):
    input_feature_names = set([x[0] for x in col_defs])
    df = generate_fake_dataframe(
        col_defs=col_defs,
    )

    origin_features = schema_to_features(df.ww.schema)
    feature_collection = my_dfs(origin_features, primitives)

    new_feature_names = (
        set([x.name for x in feature_collection.all_features]) - input_feature_names
    )
    assert new_feature_names == expected


def get_default_logical_type(tags: Set[str]):
    for tag, lt in DEFAULT_LT_FOR_TAG.items():
        if tag in tags:
            return lt

    raise Exception(f"NO DEFAULT LOGICAL TYPE FOR TAGS: {tags}")


@pytest.mark.parametrize(
    "col_defs, primitives",
    [
        (
            [
                ("idx", "Double", {"index"}),
                ("f_1", "Double"),
                ("f_2", "Double"),
                ("f_3", "Boolean"),
                ("f_4", "Boolean"),
                ("f_5", "Double"),
            ],
            [AddNumeric, Absolute, SubtractNumeric],
        ),
        (
            [
                ("idx", "Double", {"index"}),
                ("t_idx", "Datetime", {"time_index"}),
                ("f_2", "Double"),
                ("f_3", "Boolean"),
                ("f_4", "Boolean"),
                ("f_5", "Double"),
            ],
            [Lag, Day, Absolute, AddNumeric],
        ),
    ],
)
@patch.object(Feature, "_generate_hash", lambda x: x.name)
def test_compare_dfs(col_defs, primitives):
    input_feature_names = set([x[0] for x in col_defs])
    df = generate_fake_dataframe(
        col_defs=col_defs,
    )

    es = EntitySet(id="nums")
    es.add_dataframe(df, "nums", index="idx")

    features_old = dfs(
        entityset=es,
        target_dataframe_name="nums",
        trans_primitives=primitives,
        features_only=True,
    )

    origin_features = schema_to_features(df.ww.schema)
    features_collection = my_dfs(origin_features, primitives)

    feature_names_old = set([x.get_name() for x in features_old]) - input_feature_names

    feature_names_new = (
        set([x.name for x in features_collection.all_features]) - input_feature_names
    )

    assert feature_names_old == feature_names_new
