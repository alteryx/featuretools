from woodwork.logical_types import Double

from featuretools.entityset.entityset import EntitySet
from featuretools.feature_base.feature_base import (
    FeatureBase,
    IdentityFeature,
    TransformFeature,
)
from featuretools.feature_discovery.convertors import (
    convert_feature_list_to_featurebase_list,
    convert_feature_to_featurebase,
    convert_featurebase_to_feature,
)
from featuretools.feature_discovery.feature_discovery import (
    lite_dfs,
    schema_to_features,
)
from featuretools.feature_discovery.type_defs import (
    LiteFeature,
)
from featuretools.primitives import Absolute, AddNumeric, Lag
from featuretools.synthesis import dfs
from featuretools.tests.feature_discovery.test_feature_discovery import (
    TestMultiOutputPrimitive,
)
from featuretools.tests.testing_utils.generate_fake_dataframe import (
    generate_fake_dataframe,
)


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

    f1 = LiteFeature("f_1", Double)
    f2 = LiteFeature("f_2", Double)
    f3 = LiteFeature(
        name="f_1 + f_2",
        tags={"numeric"},
        primitive=AddNumeric(),
        base_features=[f1, f2],
    )

    orig_features = set([f1, f2, f3])

    assert len(orig_features.symmetric_difference(converted_features)) == 0


def test_origin_feature_to_featurebase():
    df = generate_fake_dataframe(
        col_defs=[("idx", "Double", {"index"}), ("f_1", "Double")],
    )
    # TODO(dreed): don't like how I have to make an entityset
    es = EntitySet(id="test")
    es.add_dataframe(df, df.ww.name)

    origin_features = schema_to_features(df.ww.schema)
    f_1 = [f for f in origin_features if f.name == "f_1"][0]
    fb = convert_feature_to_featurebase(f_1, df, {})

    assert isinstance(fb, IdentityFeature)
    assert fb.get_name() == "f_1"

    f_1.rename("new name")
    df.ww.rename({"f_1": "new name"}, inplace=True)
    fb = convert_feature_to_featurebase(f_1, df, {})

    assert isinstance(fb, IdentityFeature)
    assert fb.get_name() == "new name"


def test_stacked_feature_to_featurebase():
    df = generate_fake_dataframe(
        col_defs=[("idx", "Double", {"index"}), ("f_1", "Double")],
    )
    # TODO(dreed): don't like how I have to make an entityset
    es = EntitySet(id="test")
    es.add_dataframe(df, df.ww.name)

    origin_features = schema_to_features(df.ww.schema)
    f_1 = [f for f in origin_features if f.name == "f_1"][0]
    fc = lite_dfs([f_1], [Absolute()])

    f_2 = [f for f in fc.all_features if f.name == "ABSOLUTE(f_1)"][0]

    fb = convert_feature_to_featurebase(f_2, df, {})

    assert isinstance(fb, TransformFeature)
    assert fb.get_name() == "ABSOLUTE(f_1)"
    assert len(fb.base_features) == 1
    assert fb.base_features[0].get_name() == "f_1"

    f_2.rename("f_2")
    fb = convert_feature_to_featurebase(f_2, df, {})

    assert isinstance(fb, TransformFeature)
    assert fb.get_name() == "f_2"
    assert len(fb.base_features) == 1
    assert fb.base_features[0].get_name() == "f_1"


def test_multi_output_to_featurebase():
    df = generate_fake_dataframe(
        col_defs=[
            ("idx", "Double", {"index"}),
            ("f_1", "NaturalLanguage"),
        ],
    )
    # TODO(dreed): don't like how I have to make an entityset
    es = EntitySet(id="test")
    es.add_dataframe(df, df.ww.name)

    origin_features = schema_to_features(df.ww.schema)
    f_1 = [f for f in origin_features if f.name == "f_1"][0]
    fc = lite_dfs([f_1], [TestMultiOutputPrimitive()], parallelize=False)

    lsa_features = [f for f in fc.all_features if f.get_primitive_name() == "test_mo"]
    assert len(lsa_features) == 2

    # Test Single LiteFeature
    fb = convert_feature_to_featurebase(lsa_features[0], df, {})
    assert isinstance(fb, TransformFeature)
    assert fb.get_name() == "TEST_MO(f_1)"
    assert len(fb.base_features) == 1
    assert set(fb.get_feature_names()) == set(["TEST_MO(f_1)[0]", "TEST_MO(f_1)[1]"])
    assert fb.base_features[0].get_name() == "f_1"

    # Test that feature gets consolidated
    fb_list = convert_feature_list_to_featurebase_list(lsa_features, df)
    assert len(fb_list) == 1
    assert fb_list[0].get_name() == "TEST_MO(f_1)"
    assert len(fb_list[0].base_features) == 1
    assert set(fb_list[0].get_feature_names()) == set(
        ["TEST_MO(f_1)[0]", "TEST_MO(f_1)[1]"],
    )
    assert fb_list[0].base_features[0].get_name() == "f_1"

    lsa_features[0].rename("f_2")
    lsa_features[1].rename("f_3")

    fb = convert_feature_to_featurebase(lsa_features[0], df, {})
    assert isinstance(fb, TransformFeature)
    assert len(fb.base_features) == 1
    assert set(fb.get_feature_names()) == set(["f_2", "f_3"])
    assert fb.base_features[0].get_name() == "f_1"

    # Test that feature gets consolidated
    fb_list = convert_feature_list_to_featurebase_list(lsa_features, df)
    assert len(fb_list) == 1
    assert len(fb_list[0].base_features) == 1
    assert set(fb_list[0].get_feature_names()) == set(["f_2", "f_3"])
    assert fb_list[0].base_features[0].get_name() == "f_1"


def test_stacking_on_multioutput_to_featurebase():
    col_defs = [
        ("idx", "Double", {"index"}),
        ("t_idx", "Datetime", {"time_index"}),
        ("f_1", "NaturalLanguage"),
    ]
    df = generate_fake_dataframe(
        col_defs=col_defs,
    )
    # TODO(dreed): don't like how I have to make an entityset
    es = EntitySet(id="test")
    es.add_dataframe(df, df.ww.name)

    origin_features = schema_to_features(df.ww.schema)
    time_index_feature = [f for f in origin_features if f.name == "t_idx"][0]
    f_1 = [f for f in origin_features if f.name == "f_1"][0]

    fc = lite_dfs([f_1], [TestMultiOutputPrimitive()], parallelize=False)
    lsa_features = [f for f in fc.all_features if f.get_primitive_name() == "test_mo"]
    assert len(lsa_features) == 2

    fc = lite_dfs(
        lsa_features + [time_index_feature],
        [Lag(periods=2)],
        parallelize=False,
    )
    lag_features = [f for f in fc.all_features if f.get_primitive_name() == "lag"]
    assert len(lag_features) == 2

    fb_list = convert_feature_list_to_featurebase_list(lag_features, df)

    assert len(fb_list) == 2
    assert isinstance(fb_list[0], TransformFeature)
    assert set([x.get_name() for x in fb_list]) == set(
        [
            "LAG(TEST_MO(f_1)[0], t_idx, periods=2)",
            "LAG(TEST_MO(f_1)[1], t_idx, periods=2)",
        ],
    )

    lsa_features[0].rename("f_2")
    lsa_features[1].rename("f_3")
    fc = lite_dfs(lsa_features + [time_index_feature], [Lag(periods=2)])
    lag_features = [f for f in fc.all_features if f.get_primitive_name() == "lag"]
    assert len(lag_features) == 2

    fb_list = convert_feature_list_to_featurebase_list(lag_features, df)
    assert len(fb_list) == 2
    assert isinstance(fb_list[0], TransformFeature)
    assert set([x.get_name() for x in fb_list]) == set(
        ["LAG(f_2, t_idx, periods=2)", "LAG(f_3, t_idx, periods=2)"],
    )
