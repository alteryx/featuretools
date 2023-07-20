from woodwork.logical_types import Double, NaturalLanguage

from featuretools.entityset.entityset import EntitySet
from featuretools.feature_base.feature_base import (
    FeatureBase,
    IdentityFeature,
    TransformFeature,
)
from featuretools.feature_discovery.convertors import (
    _convert_feature_to_featurebase,
    convert_feature_list_to_featurebase_list,
    convert_featurebase_list_to_feature_list,
)
from featuretools.feature_discovery.feature_discovery import (
    generate_features_from_primitives,
    schema_to_features,
)
from featuretools.feature_discovery.LiteFeature import (
    LiteFeature,
)
from featuretools.primitives import Absolute, AddNumeric, Lag
from featuretools.synthesis import dfs
from featuretools.tests.feature_discovery.test_feature_discovery import (
    MultiOutputPrimitiveForTest,
)
from featuretools.tests.testing_utils.generate_fake_dataframe import (
    generate_fake_dataframe,
)


def test_convert_featurebase_list_to_feature_list():
    col_defs = [
        ("idx", "Integer", {"index"}),
        ("f_1", "Double"),
        ("f_2", "Double"),
        ("f_3", "NaturalLanguage"),
    ]

    df = generate_fake_dataframe(
        col_defs=col_defs,
    )

    es = EntitySet(id="es")
    es.add_dataframe(df, df.ww.name)

    fdefs = dfs(
        entityset=es,
        target_dataframe_name=df.ww.name,
        trans_primitives=[AddNumeric, MultiOutputPrimitiveForTest],
        features_only=True,
        max_depth=1,
    )
    assert isinstance(fdefs, list)
    assert isinstance(fdefs[0], FeatureBase)

    converted_features = set(convert_featurebase_list_to_feature_list(fdefs))

    f1 = LiteFeature("f_1", Double)
    f2 = LiteFeature("f_2", Double)
    f3 = LiteFeature("f_3", NaturalLanguage)
    fadd = LiteFeature(
        name="f_1 + f_2",
        tags={"numeric"},
        primitive=AddNumeric(),
        base_features=[f1, f2],
    )
    fmo0 = LiteFeature(
        name="TEST_MO(f_3)[0]",
        tags={"numeric"},
        primitive=MultiOutputPrimitiveForTest(),
        base_features=[f3],
        idx=0,
    )
    fmo1 = LiteFeature(
        name="TEST_MO(f_3)[1]",
        tags={"numeric"},
        primitive=MultiOutputPrimitiveForTest(),
        base_features=[f3],
        idx=1,
    )
    fmo0.related_features = {fmo1}
    fmo1.related_features = {fmo0}

    orig_features = set([f1, f2, fadd, fmo0, fmo1])

    assert len(orig_features.symmetric_difference(converted_features)) == 0


def test_origin_feature_to_featurebase():
    df = generate_fake_dataframe(
        col_defs=[("idx", "Double", {"index"}), ("f_1", "Double")],
    )
    es = EntitySet(id="test")
    es.add_dataframe(df, df.ww.name)

    origin_features = schema_to_features(df.ww.schema)
    f_1 = [f for f in origin_features if f.name == "f_1"][0]
    fb = _convert_feature_to_featurebase(f_1, df, {})

    assert isinstance(fb, IdentityFeature)
    assert fb.get_name() == "f_1"

    f_1.set_alias("new name")
    df.ww.rename({"f_1": "new name"}, inplace=True)
    fb = _convert_feature_to_featurebase(f_1, df, {})

    assert isinstance(fb, IdentityFeature)
    assert fb.get_name() == "new name"


def test_stacked_feature_to_featurebase():
    df = generate_fake_dataframe(
        col_defs=[("idx", "Double", {"index"}), ("f_1", "Double")],
    )
    es = EntitySet(id="test")
    es.add_dataframe(df, df.ww.name)

    origin_features = schema_to_features(df.ww.schema)
    f_1 = [f for f in origin_features if f.name == "f_1"][0]
    features = generate_features_from_primitives([f_1], [Absolute()])

    f_2 = [f for f in features if f.name == "ABSOLUTE(f_1)"][0]

    fb = _convert_feature_to_featurebase(f_2, df, {})

    assert isinstance(fb, TransformFeature)
    assert fb.get_name() == "ABSOLUTE(f_1)"
    assert len(fb.base_features) == 1
    assert fb.base_features[0].get_name() == "f_1"

    f_2.set_alias("f_2")
    fb = _convert_feature_to_featurebase(f_2, df, {})

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
    es = EntitySet(id="test")
    es.add_dataframe(df, df.ww.name)

    origin_features = schema_to_features(df.ww.schema)
    f_1 = [f for f in origin_features if f.name == "f_1"][0]
    features = generate_features_from_primitives([f_1], [MultiOutputPrimitiveForTest()])

    lsa_features = [f for f in features if f.get_primitive_name() == "test_mo"]
    assert len(lsa_features) == 2

    # Test Single LiteFeature
    fb = _convert_feature_to_featurebase(lsa_features[0], df, {})
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

    lsa_features[0].set_alias("f_2")
    lsa_features[1].set_alias("f_3")

    fb = _convert_feature_to_featurebase(lsa_features[0], df, {})
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
    es = EntitySet(id="test")
    es.add_dataframe(df, df.ww.name)

    origin_features = schema_to_features(df.ww.schema)
    time_index_feature = [f for f in origin_features if f.name == "t_idx"][0]
    f_1 = [f for f in origin_features if f.name == "f_1"][0]

    features = generate_features_from_primitives([f_1], [MultiOutputPrimitiveForTest()])
    lsa_features = [f for f in features if f.get_primitive_name() == "test_mo"]
    assert len(lsa_features) == 2

    features = generate_features_from_primitives(
        lsa_features + [time_index_feature],
        [Lag(periods=2)],
    )
    lag_features = [f for f in features if f.get_primitive_name() == "lag"]
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

    lsa_features[0].set_alias("f_2")
    lsa_features[1].set_alias("f_3")
    features = generate_features_from_primitives(
        lsa_features + [time_index_feature],
        [Lag(periods=2)],
    )
    lag_features = [f for f in features if f.get_primitive_name() == "lag"]
    assert len(lag_features) == 2

    fb_list = convert_feature_list_to_featurebase_list(lag_features, df)
    assert len(fb_list) == 2
    assert isinstance(fb_list[0], TransformFeature)
    assert set([x.get_name() for x in fb_list]) == set(
        ["LAG(f_2, t_idx, periods=2)", "LAG(f_3, t_idx, periods=2)"],
    )
