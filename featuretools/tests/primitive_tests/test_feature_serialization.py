import os

import pytest
from pympler.asizeof import asizeof

from ..testing_utils import make_ecommerce_entityset

import featuretools as ft
from featuretools.primitives import make_agg_primitive
from featuretools.utils.pickle_utils import save_obj_pickle
from featuretools.variable_types import Numeric


@pytest.fixture(scope='module')
def es():
    return make_ecommerce_entityset()


def test_pickle_features(es):
    features_no_pickle = ft.dfs(target_entity='sessions', entityset=es, features_only=True)

    dir_path = os.path.dirname(os.path.realpath(__file__))
    filepath = os.path.join(dir_path, 'test_feature')
    es_filepath = os.path.join(dir_path, 'test_entityset')

    # pickle entityset
    save_obj_pickle(es, es_filepath)

    ft.save_features(features_no_pickle, filepath)
    features_pickle = ft.load_features(filepath)
    for feat_1, feat_2 in zip(features_no_pickle, features_pickle):
        assert feat_1.hash() == feat_2.hash()
        assert feat_1.entityset == feat_2.entityset

    # file is smaller than entityset in memory
    assert os.path.getsize(filepath) < asizeof(es)

    # file is smaller than entityset pickled
    assert os.path.getsize(filepath) < os.path.getsize(es_filepath)
    os.remove(filepath)
    os.remove(es_filepath)


def test_pickle_features_with_custom_primitive(es):
    NewMax = make_agg_primitive(
        lambda x: max(x),
        name="NewMax",
        input_types=[Numeric],
        return_type=Numeric,
        description="Calculate means ignoring nan values")

    features_no_pickle = ft.dfs(target_entity='sessions', entityset=es,
                                agg_primitives=["Last", "Mean", NewMax], features_only=True)

    assert any([isinstance(feat.primitive, NewMax) for feat in features_no_pickle])
    dir_path = os.path.dirname(os.path.realpath(__file__))
    filepath = os.path.join(dir_path, 'test_feature')
    es_filepath = os.path.join(dir_path, 'test_entityset')

    # pickle entityset
    save_obj_pickle(es, es_filepath)

    ft.save_features(features_no_pickle, filepath)
    features_pickle = ft.load_features(filepath)
    for feat_1, feat_2 in zip(features_no_pickle, features_pickle):
        assert feat_1.hash() == feat_2.hash()
        assert feat_1.entityset == feat_2.entityset

    # file is smaller than entityset in memory
    assert os.path.getsize(filepath) < asizeof(es)

    # file is smaller than entityset pickled
    assert os.path.getsize(filepath) < os.path.getsize(es_filepath)
    os.remove(filepath)
    os.remove(es_filepath)
