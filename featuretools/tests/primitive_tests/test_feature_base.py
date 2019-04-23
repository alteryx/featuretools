import os.path

import pytest
from pympler.asizeof import asizeof

from ..testing_utils import make_ecommerce_entityset

import featuretools as ft
from featuretools import config
from featuretools.feature_base import IdentityFeature
from featuretools.primitives import Last, Mode, Sum
from featuretools.variable_types import Categorical, Datetime, Id, Numeric


@pytest.fixture(scope='module')
def es():
    return make_ecommerce_entityset()


@pytest.fixture(scope='module')
def es_numeric():
    return make_ecommerce_entityset(with_integer_time_index=True)


def test_copy_features_does_not_copy_entityset(es):
    agg = ft.Feature(es['log']['value'], parent_entity=es['sessions'], primitive=Sum)
    agg_where = ft.Feature(es['log']['value'], parent_entity=es['sessions'],
                           where=IdentityFeature(es['log']['value']) == 2, primitive=Sum)
    agg_use_previous = ft.Feature(es['log']['value'], parent_entity=es['sessions'],
                                  use_previous='4 days', primitive=Sum)
    agg_use_previous_where = ft.Feature(es['log']['value'], parent_entity=es['sessions'],
                                        where=IdentityFeature(es['log']['value']) == 2,
                                        use_previous='4 days', primitive=Sum)
    features = [agg, agg_where, agg_use_previous, agg_use_previous_where]
    in_memory_size = asizeof(locals())
    copied = [f.copy() for f in features]
    new_in_memory_size = asizeof(locals())
    assert new_in_memory_size < 2 * in_memory_size


def test_get_dependencies(es):
    f = ft.Feature(es['log']['value'])
    agg1 = ft.Feature(f, parent_entity=es['sessions'], primitive=Sum)
    agg2 = ft.Feature(agg1, parent_entity=es['customers'], primitive=Sum)
    d1 = ft.Feature(agg2, es['sessions'])
    shallow = d1.get_dependencies(deep=False, ignored=None)
    deep = d1.get_dependencies(deep=True, ignored=None)
    ignored = set([agg1.hash()])
    deep_ignored = d1.get_dependencies(deep=True, ignored=ignored)
    assert [s.hash() for s in shallow] == [agg2.hash()]
    assert [d.hash() for d in deep] == [agg2.hash(), agg1.hash(), f.hash()]
    assert [d.hash() for d in deep_ignored] == [agg2.hash()]


def test_get_depth(es):
    es = make_ecommerce_entityset()
    f = ft.Feature(es['log']['value'])
    g = ft.Feature(es['log']['value'])
    agg1 = ft.Feature(f, parent_entity=es['sessions'], primitive=Last)
    agg2 = ft.Feature(agg1, parent_entity=es['customers'], primitive=Last)
    d1 = ft.Feature(agg2, es['sessions'])
    d2 = ft.Feature(d1, es['log'])
    assert d2.get_depth() == 4
    # Make sure this works if we pass in two of the same
    # feature. This came up when user supplied duplicates
    # in the seed_features of DFS.
    assert d2.get_depth(stop_at=[f, g]) == 4
    assert d2.get_depth(stop_at=[f, g, agg1]) == 3
    assert d2.get_depth(stop_at=[f, g, agg1]) == 3
    assert d2.get_depth(stop_at=[f, g, agg2]) == 2
    assert d2.get_depth(stop_at=[f, g, d1]) == 1
    assert d2.get_depth(stop_at=[f, g, d2]) == 0


def test_squared(es):
    feature = ft.Feature(es['log']['value'])
    squared = feature * feature
    assert len(squared.base_features) == 2
    assert squared.base_features[0].hash() == squared.base_features[1].hash()


def test_return_type_inference(es):
    mode = ft.Feature(es["log"]["priority_level"], parent_entity=es["customers"], primitive=Mode)
    assert mode.variable_type == es["log"]["priority_level"].__class__


def test_return_type_inference_direct_feature(es):
    mode = ft.Feature(es["log"]["priority_level"], parent_entity=es["customers"], primitive=Mode)
    mode_session = ft.Feature(mode, es["sessions"])
    assert mode_session.variable_type == es["log"]["priority_level"].__class__


def test_return_type_inference_datetime_time_index(es):
    last = ft.Feature(es["log"]["datetime"], parent_entity=es["customers"], primitive=Last)
    assert last.variable_type == Datetime


def test_return_type_inference_numeric_time_index(es_numeric):
    last = ft.Feature(es_numeric["log"]["datetime"], parent_entity=es_numeric["customers"], primitive=Last)
    assert last.variable_type == Numeric


def test_return_type_inference_id(es):
    # direct features should keep Id variable type
    direct_id_feature = ft.Feature(es["sessions"]["customer_id"], es["log"])
    assert direct_id_feature.variable_type == Id

    # aggregations of Id variable types should get converted
    mode = ft.Feature(es["log"]["session_id"], parent_entity=es["customers"], primitive=Mode)
    assert mode.variable_type == Categorical

    # also test direct feature of aggregation
    mode_direct = ft.Feature(mode, es["sessions"])
    assert mode_direct.variable_type == Categorical


def test_set_data_path(es):
    key = "primitive_data_folder"

    # Don't change orig_path
    orig_path = config.get(key)
    new_path = "/example/new/directory"
    filename = "test.csv"

    # Test that default path works
    sum_prim = Sum()
    assert sum_prim.get_filepath(filename) == os.path.join(orig_path, filename)

    # Test that new path works
    config.set({key: new_path})
    assert sum_prim.get_filepath(filename) == os.path.join(new_path, filename)

    # Test that new path with trailing / works
    new_path += "/"
    config.set({key: new_path})
    assert sum_prim.get_filepath(filename) == os.path.join(new_path, filename)

    # Test that the path is correct on newly defined feature
    sum_prim2 = Sum()
    assert sum_prim2.get_filepath(filename) == os.path.join(new_path, filename)

    # Ensure path was reset
    config.set({key: orig_path})
    assert config.get(key) == orig_path
