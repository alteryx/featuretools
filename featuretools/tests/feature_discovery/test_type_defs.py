import json

from woodwork.logical_types import Double

from featuretools.entityset.entityset import EntitySet
from featuretools.feature_base.feature_base import FeatureBase
from featuretools.feature_discovery.type_defs import Feature, convert_old_to_new
from featuretools.primitives import AddNumeric
from featuretools.synthesis import dfs
from featuretools.tests.testing_utils.generate_fake_dataframe import (
    generate_fake_dataframe,
)


def test_feature_type():

    Feature(
        name="Column 1",
        primitive=AddNumeric,
        logical_type=Double,
    )


def test_feature_to_dict():
    f1 = Feature("f1", Double)
    f2 = Feature("f2", Double)
    f = Feature(
        name="Column 1",
        primitive=AddNumeric,
        logical_type=Double,
        base_features=[f1, f2],
    )

    expected = {
        "name": "Column 1",
        "logical_type": "Double",
        "tags": [],
        "primitive": "AddNumeric",
        "base_features": [
            {
                "name": "f1",
                "logical_type": "Double",
                "tags": [],
                "primitive": None,
                "base_features": [],
                "df_id": None,
                "id": "3f524cdc07a11d7c6220bdb049fe8dd41b27483c96cc59b581e022d547290d69",
            },
            {
                "name": "f2",
                "logical_type": "Double",
                "tags": [],
                "primitive": None,
                "base_features": [],
                "df_id": None,
                "id": "e4ab4e3b1493d5a997b4e51cdefbaa10570ef3ea9432bd72e7b6a89654ceb7f6",
            },
        ],
        "df_id": None,
        "id": "b999e3fdddba9cfca2d63fe4030332210578f07698f21e82d3c91c094b1862cf",
    }

    actual = f.to_dict()
    json_str = json.dumps(actual)
    assert actual == expected
    assert json.dumps(expected) == json_str


def test_feature_from_dict():
    f1 = Feature("f1", Double)
    f2 = Feature("f2", Double)
    f_orig = Feature(
        name="Column 1",
        primitive=AddNumeric,
        logical_type=Double,
        base_features=[f1, f2],
    )

    input_dict = f_orig.to_dict()
    f_from_dict = Feature.from_dict(input_dict)
    assert f_orig == f_from_dict


def test_convert_old_to_new():
    col_defs = [
        ("f_1", "Double"),
        ("f_2", "Double"),
    ]

    df = generate_fake_dataframe(
        col_defs=col_defs,
        include_index=True,
    )

    es = EntitySet(id="nums")
    es.add_dataframe(df, "nums", index="idx")

    fdefs = dfs(
        entityset=es,
        target_dataframe_name="nums",
        trans_primitives=[AddNumeric],
        features_only=True,
    )
    assert isinstance(fdefs, list)
    assert isinstance(fdefs[0], FeatureBase)

    convert_old_to_new(fdefs[2])

    converted_features = set([convert_old_to_new(x) for x in fdefs])

    f1 = Feature("f_1", Double)
    f2 = Feature("f_2", Double)
    f3 = Feature(
        name="f_1 + f_2",
        tags={"numeric"},
        primitive=AddNumeric,
        base_features=[f1, f2],
    )

    orig_features = set([f1, f2, f3])

    assert len(orig_features.symmetric_difference(converted_features)) == 0
