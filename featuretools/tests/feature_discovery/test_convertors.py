from typing import List, Type

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
from featuretools.feature_discovery.feature_discovery import my_dfs, schema_to_features
from featuretools.feature_discovery.type_defs import (
    Feature,
)
from featuretools.primitives import LSA, Absolute, AddNumeric, Lag
from featuretools.primitives.base.primitive_base import PrimitiveBase
from featuretools.synthesis import dfs
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


# def test_lag_on_lsa():
#     col_defs = [
#         ("idx", "Double", {"index"}),
#         ("t_idx", "Datetime", {"time_index"}),
#         ("f_1", "NaturalLanguage"),
#     ]
#     df = generate_fake_dataframe(
#         col_defs=col_defs,
#     )

#     primitives = [LSA]

#     es = EntitySet(id="test")
#     es.add_dataframe(df, df.ww.name, index="idx")

#     features_old = dfs(
#         entityset=es,
#         target_dataframe_name=df.ww.name,
#         trans_primitives=primitives,
#         features_only=True,
#     )
#     assert len(features_old) == 1

#     # Generate Old LSA Feature
#     lsa_feature_old = features_old[0]
#     time_index_feature = OldFeature(df.ww["t_idx"])

#     base_features = [lsa_feature_old, time_index_feature]

#     lag_instance = Lag(periods=2)

#     # Assert that stacking on Multi-output fails as expected
#     with pytest.raises(ValueError, match="Cannot stack on whole multi-output feature."):
#         OldFeature(base_features, primitive=lag_instance)

#     # Manually Create Stack features, by splitting multi-output feature
#     lsa_feature_list_old = [lsa_feature_old[i] for i in range(lsa_feature_old.number_output_features)]
#     lagged_lsa_features = []
#     for f in lsa_feature_list_old:
#         base_features = [f, time_index_feature]
#         lagged_lsa_features.append(OldFeature(base_features, primitive=lag_instance))


#     # With New DFS
#     origin_features = schema_to_features(df.ww.schema)
#     new_features = my_dfs(origin_features, [LSA()])

#     lsa_features = [f for f in new_features.all_features if f.get_primitive_name() == 'lsa']
#     assert len(lsa_features) == 2


#     converted_features = convert_feature_list_to_featurebase_list(lsa_features, df)

#     assert len(converted_features) == 1
#     lsa_feature = converted_features[0]
#     assert lsa_feature.get_feature_names() == ["LSA(f_1)[0]", "LSA(f_1)[1]"]

#     raise


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
    fc = my_dfs([f_1], [Absolute()])

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

    # primitive_instance = LSA()
    origin_features = schema_to_features(df.ww.schema)
    f_1 = [f for f in origin_features if f.name == "f_1"][0]
    fc = my_dfs([f_1], [LSA()])

    lsa_features = [f for f in fc.all_features if f.get_primitive_name() == "lsa"]
    assert len(lsa_features) == 2

    # Test Single Feature
    fb = convert_feature_to_featurebase(lsa_features[0], df, {})
    assert isinstance(fb, TransformFeature)
    assert fb.get_name() == "LSA(f_1)"
    assert len(fb.base_features) == 1
    assert set(fb.get_feature_names()) == set(["LSA(f_1)[0]", "LSA(f_1)[1]"])
    assert fb.base_features[0].get_name() == "f_1"

    # Test that feature gets consolidated
    fb_list = convert_feature_list_to_featurebase_list(lsa_features, df)
    assert len(fb_list) == 1
    assert fb_list[0].get_name() == "LSA(f_1)"
    assert len(fb_list[0].base_features) == 1
    assert set(fb_list[0].get_feature_names()) == set(["LSA(f_1)[0]", "LSA(f_1)[1]"])
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

    fc = my_dfs([f_1], [LSA()])
    lsa_features = [f for f in fc.all_features if f.get_primitive_name() == "lsa"]
    assert len(lsa_features) == 2

    fc = my_dfs(lsa_features + [time_index_feature], [Lag(periods=2)])
    lag_features = [f for f in fc.all_features if f.get_primitive_name() == "lag"]
    assert len(lag_features) == 2

    fb_list = convert_feature_list_to_featurebase_list(lag_features, df)

    assert len(fb_list) == 2
    assert isinstance(fb_list[0], TransformFeature)
    assert set([x.get_name() for x in fb_list]) == set(
        ["LAG(LSA(f_1)[0], t_idx, periods=2)", "LAG(LSA(f_1)[1], t_idx, periods=2)"],
    )

    lsa_features[0].rename("f_2")
    lsa_features[1].rename("f_3")
    fc = my_dfs(lsa_features + [time_index_feature], [Lag(periods=2)])
    lag_features = [f for f in fc.all_features if f.get_primitive_name() == "lag"]
    assert len(lag_features) == 2

    fb_list = convert_feature_list_to_featurebase_list(lag_features, df)
    assert len(fb_list) == 2
    assert isinstance(fb_list[0], TransformFeature)
    assert set([x.get_name() for x in fb_list]) == set(
        ["LAG(f_2, t_idx, periods=2)", "LAG(f_3, t_idx, periods=2)"],
    )


def test_convert_feature_to_feature_base():
    col_defs = [
        ("idx", "Integer", {"index"}),
        ("f_1", "Double"),
        ("f_2", "Double"),
    ]

    df = generate_fake_dataframe(
        col_defs=col_defs,
    )

    es = EntitySet(id="nums")
    es.add_dataframe(df, "nums", index="idx")

    primitives: List[Type[PrimitiveBase]] = [Absolute]

    dfs(
        entityset=es,
        target_dataframe_name="nums",
        trans_primitives=primitives,
        features_only=True,
    )

    origin_features = schema_to_features(df.ww.schema)

    features_collection = my_dfs(origin_features, primitives)

    features_new = [
        x for x in features_collection.all_features if "index" not in x.tags
    ]

    convert_feature_list_to_featurebase_list(df, features_new)

    raise

    # f1 = set(FeaturesSerializer(features_old).to_dict()["feature_list"])

    # f2 = set(FeaturesSerializer(converted_features).to_dict()["feature_list"])

    # assert f1 == f2
