from unittest.mock import patch

import pytest
from woodwork.column_schema import ColumnSchema
from woodwork.logical_types import (
    Boolean,
    BooleanNullable,
    Datetime,
    Double,
    NaturalLanguage,
    Ordinal,
)

from featuretools.entityset.entityset import EntitySet
from featuretools.feature_discovery.feature_discovery import (
    _get_features,
    _get_matching_features,
    _index_column_set,
    generate_features_from_primitives,
    schema_to_features,
)
from featuretools.feature_discovery.FeatureCollection import FeatureCollection
from featuretools.feature_discovery.LiteFeature import (
    LiteFeature,
)
from featuretools.feature_discovery.utils import column_schema_to_keys
from featuretools.primitives import (
    Absolute,
    AddNumeric,
    Count,
    DateFirstEvent,
    Equal,
    Lag,
    MultiplyNumericBoolean,
    NumUnique,
    TransformPrimitive,
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


class MultiOutputPrimitiveForTest(TransformPrimitive):
    name = "test_mo"
    input_types = [ColumnSchema(logical_type=NaturalLanguage)]
    return_type = ColumnSchema(semantic_tags={"numeric"})
    number_output_features = 2


class DoublePrimitiveForTest(TransformPrimitive):
    name = "test_double"
    input_types = [ColumnSchema(logical_type=Double)]
    return_type = ColumnSchema(logical_type=Double)


@pytest.mark.parametrize(
    "column_schema, expected",
    [
        (ColumnSchema(logical_type=Double), "Double"),
        (ColumnSchema(semantic_tags={"index"}), "index"),
        (
            ColumnSchema(logical_type=Double, semantic_tags={"index", "other"}),
            "Double,index,other",
        ),
    ],
)
def test_column_schema_to_keys(column_schema, expected):
    actual = column_schema_to_keys(column_schema)
    assert set(actual) == set(expected)


@pytest.mark.parametrize(
    "column_list, expected",
    [
        ([ColumnSchema(logical_type=Boolean)], [("Boolean", 1)]),
        ([ColumnSchema()], [("ANY", 1)]),
        (
            [
                ColumnSchema(logical_type=Boolean),
                ColumnSchema(logical_type=Boolean),
            ],
            [("Boolean", 2)],
        ),
    ],
)
def test_index_input_set(column_list, expected):
    actual = _index_column_set(column_list)

    assert actual == expected


@pytest.mark.parametrize(
    "feature_args, input_set, commutative, expected",
    [
        (
            [("f1", Boolean), ("f2", Boolean), ("f3", Boolean)],
            [ColumnSchema(logical_type=Boolean)],
            False,
            [["f1"], ["f2"], ["f3"]],
        ),
        (
            [("f1", Boolean), ("f2", Boolean)],
            [ColumnSchema(logical_type=Boolean), ColumnSchema(logical_type=Boolean)],
            False,
            [["f1", "f2"], ["f2", "f1"]],
        ),
        (
            [("f1", Boolean), ("f2", Boolean)],
            [ColumnSchema(logical_type=Boolean), ColumnSchema(logical_type=Boolean)],
            True,
            [["f1", "f2"]],
        ),
        (
            [("f1", Datetime, {"time_index"})],
            [ColumnSchema(logical_type=Datetime, semantic_tags={"time_index"})],
            False,
            [["f1"]],
        ),
        (
            [("f1", Double, {"other", "index"})],
            [ColumnSchema(logical_type=Double, semantic_tags={"index", "other"})],
            False,
            [["f1"]],
        ),
        (
            [
                ("f1", Double),
                ("f2", Boolean),
                ("f3", Double),
                ("f4", Boolean),
                ("f5", Double),
            ],
            [
                ColumnSchema(logical_type=Double),
                ColumnSchema(logical_type=Double),
                ColumnSchema(logical_type=Boolean),
            ],
            True,
            [
                ["f1", "f3", "f2"],
                ["f1", "f3", "f4"],
                ["f1", "f5", "f2"],
                ["f1", "f5", "f4"],
                ["f3", "f5", "f2"],
                ["f3", "f5", "f4"],
            ],
        ),
    ],
)
@patch.object(LiteFeature, "_generate_hash", lambda x: x.name)
def test_get_features(feature_args, input_set, commutative, expected):
    features = [LiteFeature(*args) for args in feature_args]
    feature_collection = FeatureCollection(features).reindex()

    column_keys = _index_column_set(input_set)
    actual = _get_features(feature_collection, tuple(column_keys), commutative)

    assert set([tuple([y.id for y in x]) for x in actual]) == set(
        [tuple(x) for x in expected],
    )


@pytest.mark.parametrize(
    "feature_args, primitive, expected",
    [
        (
            [("f1", Double), ("f2", Double), ("f3", Double)],
            AddNumeric,
            [["f1", "f2"], ["f1", "f3"], ["f2", "f3"]],
        ),
        (
            [("f1", Boolean), ("f2", Boolean), ("f3", Boolean)],
            AddNumeric,
            [],
        ),
        (
            [("f7", Double), ("f8", Boolean)],
            MultiplyNumericBoolean,
            [["f7", "f8"]],
        ),
        (
            [("f9", Datetime)],
            DateFirstEvent,
            [],
        ),
        (
            [("f10", Datetime, {"time_index"})],
            DateFirstEvent,
            [["f10"]],
        ),
        (
            [("f11", Datetime, {"time_index"}), ("f12", Double)],
            NumUnique,
            [],
        ),
        (
            [("f13", Datetime, {"time_index"}), ("f14", Double), ("f15", Ordinal)],
            NumUnique,
            [["f15"]],
        ),
        (
            [("f16", Datetime, {"time_index"}), ("f17", Double), ("f18", Ordinal)],
            Equal,
            [["f16", "f17"], ["f16", "f18"], ["f17", "f18"]],
        ),
        (
            [
                ("t_idx", Datetime, {"time_index"}),
                ("f19", Ordinal),
                ("f20", Double),
                ("f21", Boolean),
                ("f22", BooleanNullable),
            ],
            Lag,
            [["f19", "t_idx"], ["f20", "t_idx"], ["f21", "t_idx"], ["f22", "t_idx"]],
        ),
        (
            [
                ("idx", Double, {"index"}),
                ("f23", Double),
            ],
            Count,
            [["idx"]],
        ),
        (
            [
                ("idx", Double, {"index"}),
                ("f23", Double),
            ],
            AddNumeric,
            [],
        ),
    ],
)
@patch.object(LiteFeature, "__lt__", lambda x, y: x.name < y.name)
def test_get_matching_features(feature_args, primitive, expected):
    features = [LiteFeature(*args) for args in feature_args]
    feature_collection = FeatureCollection(features).reindex()
    actual = _get_matching_features(feature_collection, primitive())
    assert [[y.name for y in x] for x in actual] == expected


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
@patch.object(LiteFeature, "__lt__", lambda x, y: x.name < y.name)
def test_generate_features_from_primitives(col_defs, primitives, expected):
    input_feature_names = set([x[0] for x in col_defs])
    df = generate_fake_dataframe(
        col_defs=col_defs,
    )

    origin_features = schema_to_features(df.ww.schema)
    features = generate_features_from_primitives(origin_features, primitives)

    new_feature_names = set([x.name for x in features]) - input_feature_names
    assert new_feature_names == expected


ALL_TRANSFORM_PRIMITIVES = list(get_transform_primitives().values())


@pytest.mark.parametrize(
    "col_defs, primitives",
    [
        (
            [
                ("idx", "Double", {"index"}),
                ("t_idx", "Datetime", {"time_index"}),
                ("f_3", "Boolean"),
                ("f_4", "Boolean"),
                ("f_5", "BooleanNullable"),
                ("f_6", "BooleanNullable"),
                ("f_7", "Categorical"),
                ("f_8", "Categorical"),
                ("f_9", "Datetime"),
                ("f_10", "Datetime"),
                ("f_11", "Double"),
                ("f_12", "Double"),
                ("f_13", "Integer"),
                ("f_14", "Integer"),
                ("f_15", "IntegerNullable"),
                ("f_16", "IntegerNullable"),
                ("f_17", "EmailAddress"),
                ("f_18", "EmailAddress"),
                ("f_19", "LatLong"),
                ("f_20", "LatLong"),
                ("f_21", "NaturalLanguage"),
                ("f_22", "NaturalLanguage"),
                ("f_23", "Ordinal"),
                ("f_24", "Ordinal"),
                ("f_25", "URL"),
                ("f_26", "URL"),
                ("f_27", "PostalCode"),
                ("f_28", "PostalCode"),
            ],
            ALL_TRANSFORM_PRIMITIVES,
        ),
    ],
)
@patch.object(LiteFeature, "_generate_hash", lambda x: x.name)
def test_compare_dfs(col_defs, primitives):
    input_feature_names = set([x[0] for x in col_defs])
    df = generate_fake_dataframe(
        col_defs=col_defs,
    )

    es = EntitySet(id="test")
    es.add_dataframe(df, "df")

    features_old = dfs(
        entityset=es,
        target_dataframe_name="df",
        trans_primitives=primitives,
        features_only=True,
        return_types="all",
        max_depth=1,
    )

    origin_features = schema_to_features(df.ww.schema)
    features = generate_features_from_primitives(origin_features, primitives)

    feature_names_old = set([x.get_name() for x in features_old]) - input_feature_names  # type: ignore

    feature_names_new = set([x.name for x in features]) - input_feature_names
    assert feature_names_old == feature_names_new


def test_generate_features_from_primitives_inputs():
    f1 = LiteFeature("f1", Double)
    with pytest.raises(
        ValueError,
        match="input_features must be an iterable of LiteFeature objects",
    ):
        generate_features_from_primitives(f1, [Absolute])

    with pytest.raises(
        ValueError,
        match="input_features must be an iterable of LiteFeature objects",
    ):
        generate_features_from_primitives([f1, "other"], [Absolute])

    with pytest.raises(
        ValueError,
        match="primitives must be a list of Primitive classes or Primitive instances",
    ):
        generate_features_from_primitives([f1], ["absolute"])

    with pytest.raises(
        ValueError,
        match="primitives must be a list of Primitive classes or Primitive instances",
    ):
        generate_features_from_primitives([f1], Absolute)
