
import os

from pympler.asizeof import asizeof

import featuretools as ft
from featuretools.primitives import make_agg_primitive
from featuretools.variable_types import Numeric


def test_pickle_features(es):
    features_original = ft.dfs(target_entity='sessions', entityset=es, features_only=True)

    dir_path = os.path.dirname(os.path.realpath(__file__))
    filepath = os.path.join(dir_path, 'test_feature')

    ft.save_features(features_original, filepath)
    features_deserializedA = ft.load_features(filepath)
    assert os.path.getsize(filepath) < asizeof(es)
    os.remove(filepath)

    with open(filepath, "w") as f:
        ft.save_features(features_original, f)
        f.close()
    features_deserializedB = ft.load_features(open(filepath))
    assert os.path.getsize(filepath) < asizeof(es)
    os.remove(filepath)

    features = ft.save_features(features_original)
    features_deserializedC = ft.load_features(features)
    assert asizeof(features) < asizeof(es)

    features_deserialized_options = [features_deserializedA, features_deserializedB, features_deserializedC]
    for features_deserialized in features_deserialized_options:
        for feat_1, feat_2 in zip(features_original, features_deserialized):
            assert feat_1.unique_name() == feat_2.unique_name()
            assert feat_1.entityset == feat_2.entityset

def test_pickle_features_with_custom_primitive(es):
    NewMax = make_agg_primitive(
        lambda x: max(x),
        name="NewMax",
        input_types=[Numeric],
        return_type=Numeric,
        description="Calculate means ignoring nan values")

    features_original = ft.dfs(target_entity='sessions', entityset=es,
                               agg_primitives=["Last", "Mean", NewMax], features_only=True)

    assert any([isinstance(feat.primitive, NewMax) for feat in features_original])
    dir_path = os.path.dirname(os.path.realpath(__file__))
    filepath = os.path.join(dir_path, 'test_feature')

    ft.save_features(features_original, filepath)
    features_deserializedA = ft.load_features(filepath)
    assert os.path.getsize(filepath) < asizeof(es)
    os.remove(filepath)

    with open(filepath, "w") as f:
        ft.save_features(features_original, f)
        f.close()
    features_deserializedB = ft.load_features(open(filepath))
    assert os.path.getsize(filepath) < asizeof(es)
    os.remove(filepath)

    features = ft.save_features(features_original)
    features_deserializedC = ft.load_features(features)
    assert asizeof(features) < asizeof(es)

    features_deserialized_options = [features_deserializedA, features_deserializedB, features_deserializedC]
    for features_deserialized in features_deserialized_options:
        for feat_1, feat_2 in zip(features_original, features_deserialized):
            assert feat_1.unique_name() == feat_2.unique_name()
            assert feat_1.entityset == feat_2.entityset