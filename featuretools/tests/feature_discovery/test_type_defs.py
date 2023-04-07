import json
from unittest.mock import patch

import pytest
from woodwork.logical_types import Double

from featuretools.entityset.entityset import EntitySet
from featuretools.feature_base.feature_base import FeatureBase
from featuretools.feature_discovery.type_defs import (
    Feature,
    convert_featurebase_to_feature,
)
from featuretools.primitives import AddNumeric, DivideNumeric
from featuretools.primitives.standard.transform.time_series.lag import Lag
from featuretools.synthesis import dfs
from featuretools.tests.testing_utils.generate_fake_dataframe import (
    generate_fake_dataframe,
)


def test_feature_type_equality():
    f1 = Feature("f1", Double)
    f2 = Feature("f2", Double)

    # Add Numeric is Commutative, so should all be equal
    f3 = Feature(
        name="Column 1",
        primitive=AddNumeric(),
        logical_type=Double,
        base_features=[f1, f2],
    )

    f4 = Feature(
        name="Column 10",
        primitive=AddNumeric(),
        logical_type=Double,
        base_features=[f1, f2],
    )

    f5 = Feature(
        name="Column 20",
        primitive=AddNumeric(),
        logical_type=Double,
        base_features=[f2, f1],
    )

    assert f3 == f4 == f5

    # Divide Numeric is not Commutative, so should not be equal
    f6 = Feature(
        name="Column 1",
        primitive=DivideNumeric(),
        logical_type=Double,
        base_features=[f1, f2],
    )

    f7 = Feature(
        name="Column 1",
        primitive=DivideNumeric(),
        logical_type=Double,
        base_features=[f2, f1],
    )

    assert f6 != f7


def test_feature_type_assertions():
    with pytest.raises(
        AssertionError,
        match="there must be base features if give a primitive",
    ):
        Feature(
            name="Column 1",
            primitive=AddNumeric(),
            logical_type=Double,
        )


@patch.object(Feature, "_generate_hash", lambda x: x.name)
def test_feature_to_dict():
    f1 = Feature("f1", Double)
    f2 = Feature("f2", Double)
    f = Feature(
        name="Column 1",
        primitive=AddNumeric(),
        logical_type=Double,
        base_features=[f1, f2],
    )

    expected = {
        "name": "Column 1",
        "logical_type": "Double",
        "tags": ["numeric"],
        "primitive": {
            "type": "AddNumeric",
            "module": "featuretools.primitives.standard.transform.binary.add_numeric",
            "arguments": {},
        },
        "base_features": [
            {
                "name": "f1",
                "logical_type": "Double",
                "tags": ["numeric"],
                "primitive": None,
                "base_features": [],
                "df_id": None,
                "id": "f1",
            },
            {
                "name": "f2",
                "logical_type": "Double",
                "tags": ["numeric"],
                "primitive": None,
                "base_features": [],
                "df_id": None,
                "id": "f2",
            },
        ],
        "df_id": None,
        "id": "Column 1",
    }

    actual = f.to_dict()
    json_str = json.dumps(actual)
    assert actual == expected
    assert json.dumps(expected) == json_str


def test_feature_from_dict():
    f1 = Feature("f1", Double)
    f2 = Feature("f2", Double)
    f_orig = Feature(
        primitive=AddNumeric(),
        logical_type=Double,
        base_features=[f1, f2],
    )

    input_dict = f_orig.to_dict()
    f_from_dict = Feature.from_dict(input_dict)
    assert f_orig == f_from_dict


def test_feature_hash():
    bf = Feature("bf", Double)

    p1 = Lag(periods=1)
    p2 = Lag(periods=2)
    f1 = Feature(
        primitive=p1,
        logical_type=Double,
        base_features=[bf],
    )

    f2 = Feature(
        primitive=p2,
        logical_type=Double,
        base_features=[bf],
    )

    f3 = Feature(
        primitive=p2,
        logical_type=Double,
        base_features=[bf],
    )

    assert f1 != f2
    assert f2 == f3


def test_feature_forced_name():
    bf = Feature("bf", Double)

    p1 = Lag(periods=1)
    f1 = Feature(
        name="target_delay_1",
        primitive=p1,
        logical_type=Double,
        base_features=[bf],
    )
    assert f1.name == "target_delay_1"


def test_convert_featurebase_to_feature():
    col_defs = [
        ("idx", "Integer", {"index"}),
        ("f_1", "Double"),
        ("f_2", "Double"),
    ]

    df = generate_fake_dataframe(
        col_defs=col_defs,
    )

    es = EntitySet(id="es")
    es.add_dataframe(df, df.ww.name, index="idx")

    fdefs = dfs(
        entityset=es,
        target_dataframe_name=df.ww.name,
        trans_primitives=[AddNumeric],
        features_only=True,
    )
    assert isinstance(fdefs, list)
    assert isinstance(fdefs[0], FeatureBase)

    converted_features = set([convert_featurebase_to_feature(x) for x in fdefs])

    f1 = Feature("f_1", Double)
    f2 = Feature("f_2", Double)
    f3 = Feature(
        name="f_1 + f_2",
        tags={"numeric"},
        primitive=AddNumeric(),
        base_features=[f1, f2],
    )

    orig_features = set([f1, f2, f3])

    assert len(orig_features.symmetric_difference(converted_features)) == 0


# def test_convert_feature_to_feature_base():
#     col_defs = [
#         ("idx", "Integer", {"index"}),
#         ("f_1", "Double"),
#         ("f_2", "Double"),
#     ]

#     df = generate_fake_dataframe(
#         col_defs=col_defs,
#     )

#     es = EntitySet(id="nums")
#     es.add_dataframe(df, "nums", index="idx")

#     primitives: List[Type[PrimitiveBase]] = [Absolute]

#     features_old = dfs(
#         entityset=es,
#         target_dataframe_name="nums",
#         trans_primitives=primitives,
#         features_only=True,
#     )

#     origin_features = schema_to_features(df.ww.schema)

#     features_collection = my_dfs(origin_features, primitives)

#     features_new = [
#         x for x in features_collection.all_features if "index" not in x.tags
#     ]

#     converted_features = convert_feature_list_to_featurebase_list(df, features_new)

#     f1 = set(FeaturesSerializer(features_old).to_dict()["feature_list"])

#     f2 = set(FeaturesSerializer(converted_features).to_dict()["feature_list"])

#     assert f1 == f2
