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
    features_deserialized = ft.load_features(filepath)
    for feat_1, feat_2 in zip(features_original, features_deserialized):
        assert feat_1.unique_name() == feat_2.unique_name()
        assert feat_1.entityset == feat_2.entityset

    # file is smaller than entityset in memory
    assert os.path.getsize(filepath) < asizeof(es)

    os.remove(filepath)


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
    features_deserialized = ft.load_features(filepath)
    for feat_1, feat_2 in zip(features_original, features_deserialized):
        assert feat_1.unique_name() == feat_2.unique_name()
        assert feat_1.entityset == feat_2.entityset

    # file is smaller than entityset in memory
    assert os.path.getsize(filepath) < asizeof(es)

    os.remove(filepath)
