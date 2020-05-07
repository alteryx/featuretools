import os.path

import pytest
from pympler.asizeof import asizeof

import featuretools as ft
from featuretools import config
from featuretools.feature_base import IdentityFeature
from featuretools.primitives import (
    Diff,
    Last,
    Mode,
    NMostCommon,
    NumUnique,
    Sum,
    TransformPrimitive
)
from featuretools.variable_types import Categorical, Datetime, Id, Numeric


def test_copy_features_does_not_copy_entityset(pd_es):
    agg = ft.Feature(pd_es['log']['value'], parent_entity=pd_es['sessions'], primitive=Sum)
    agg_where = ft.Feature(pd_es['log']['value'], parent_entity=pd_es['sessions'],
                           where=IdentityFeature(pd_es['log']['value']) == 2, primitive=Sum)
    agg_use_previous = ft.Feature(pd_es['log']['value'], parent_entity=pd_es['sessions'],
                                  use_previous='4 days', primitive=Sum)
    agg_use_previous_where = ft.Feature(pd_es['log']['value'], parent_entity=pd_es['sessions'],
                                        where=IdentityFeature(pd_es['log']['value']) == 2,
                                        use_previous='4 days', primitive=Sum)
    features = [agg, agg_where, agg_use_previous, agg_use_previous_where]
    in_memory_size = asizeof(locals())
    copied = [f.copy() for f in features]
    new_in_memory_size = asizeof(locals())
    assert new_in_memory_size < 2 * in_memory_size


def test_get_dependencies(pd_es):
    f = ft.Feature(pd_es['log']['value'])
    agg1 = ft.Feature(f, parent_entity=pd_es['sessions'], primitive=Sum)
    agg2 = ft.Feature(agg1, parent_entity=pd_es['customers'], primitive=Sum)
    d1 = ft.Feature(agg2, pd_es['sessions'])
    shallow = d1.get_dependencies(deep=False, ignored=None)
    deep = d1.get_dependencies(deep=True, ignored=None)
    ignored = set([agg1.unique_name()])
    deep_ignored = d1.get_dependencies(deep=True, ignored=ignored)
    assert [s.unique_name() for s in shallow] == [agg2.unique_name()]
    assert [d.unique_name() for d in deep] == [agg2.unique_name(), agg1.unique_name(), f.unique_name()]
    assert [d.unique_name() for d in deep_ignored] == [agg2.unique_name()]


def test_get_depth(pd_es):
    f = ft.Feature(pd_es['log']['value'])
    g = ft.Feature(pd_es['log']['value'])
    agg1 = ft.Feature(f, parent_entity=pd_es['sessions'], primitive=Last)
    agg2 = ft.Feature(agg1, parent_entity=pd_es['customers'], primitive=Last)
    d1 = ft.Feature(agg2, pd_es['sessions'])
    d2 = ft.Feature(d1, pd_es['log'])
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


def test_squared(pd_es):
    feature = ft.Feature(pd_es['log']['value'])
    squared = feature * feature
    assert len(squared.base_features) == 2
    assert squared.base_features[0].unique_name() == squared.base_features[1].unique_name()


def test_return_type_inference(pd_es):
    mode = ft.Feature(pd_es["log"]["priority_level"], parent_entity=pd_es["customers"], primitive=Mode)
    assert mode.variable_type == pd_es["log"]["priority_level"].__class__


def test_return_type_inference_direct_feature(pd_es):
    mode = ft.Feature(pd_es["log"]["priority_level"], parent_entity=pd_es["customers"], primitive=Mode)
    mode_session = ft.Feature(mode, pd_es["sessions"])
    assert mode_session.variable_type == pd_es["log"]["priority_level"].__class__


def test_return_type_inference_index(pd_es):
    last = ft.Feature(pd_es["log"]["id"], parent_entity=pd_es["customers"], primitive=Last)
    assert last.variable_type == Categorical


def test_return_type_inference_datetime_time_index(pd_es):
    last = ft.Feature(pd_es["log"]["datetime"], parent_entity=pd_es["customers"], primitive=Last)
    assert last.variable_type == Datetime


def test_return_type_inference_numeric_time_index(int_es):
    last = ft.Feature(int_es["log"]["datetime"], parent_entity=int_es["customers"], primitive=Last)
    assert last.variable_type == Numeric


def test_return_type_inference_id(pd_es):
    # direct features should keep Id variable type
    direct_id_feature = ft.Feature(pd_es["sessions"]["customer_id"], pd_es["log"])
    assert direct_id_feature.variable_type == Id

    # aggregations of Id variable types should get converted
    mode = ft.Feature(pd_es["log"]["session_id"], parent_entity=pd_es["customers"], primitive=Mode)
    assert mode.variable_type == Categorical

    # also test direct feature of aggregation
    mode_direct = ft.Feature(mode, pd_es["sessions"])
    assert mode_direct.variable_type == Categorical


def test_set_data_path(pd_es):
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


def test_to_dictionary(pd_es):
    direct_feature = ft.Feature(pd_es["sessions"]["customer_id"], pd_es["log"])
    expected = {
        'type': 'DirectFeature',
        'dependencies': [feat.unique_name() for feat in direct_feature.get_dependencies()],
        'arguments': direct_feature.get_arguments()
    }
    assert expected == direct_feature.to_dictionary()


def test_multi_output_base_error_agg(pd_es):
    three_common = NMostCommon(3)
    tc = ft.Feature(pd_es['log']['product_id'], parent_entity=pd_es["sessions"], primitive=three_common)
    error_text = "Cannot stack on whole multi-output feature."
    with pytest.raises(ValueError, match=error_text):
        ft.Feature(tc, parent_entity=pd_es['customers'], primitive=NumUnique)


def test_multi_output_base_error_trans(pd_es):
    class TestTime(TransformPrimitive):
        name = "test_time"
        input_types = [Datetime]
        return_type = Numeric
        number_output_features = 6

    tc = ft.Feature(pd_es['customers']['date_of_birth'], primitive=TestTime)

    error_text = "Cannot stack on whole multi-output feature."
    with pytest.raises(ValueError, match=error_text):
        ft.Feature(tc, primitive=Diff)


def test_multi_output_attributes(pd_es):
    tc = ft.Feature(pd_es['log']['product_id'], parent_entity=pd_es["sessions"], primitive=NMostCommon)

    assert tc.generate_name() == 'N_MOST_COMMON(log.product_id)'
    assert tc.number_output_features == 3
    assert tc.base_features == ['<Feature: product_id>']

    assert tc[0].generate_name() == 'N_MOST_COMMON(log.product_id)[0]'
    assert tc[0].number_output_features == 1
    assert tc[0].base_features == [tc]
    assert tc.relationship_path == tc[0].relationship_path


def test_multi_output_index_error(pd_es):
    error_text = "can only access slice of multi-output feature"
    three_common = ft.Feature(pd_es['log']['product_id'],
                              parent_entity=pd_es["sessions"],
                              primitive=NMostCommon)

    with pytest.raises(AssertionError, match=error_text):
        single = ft.Feature(pd_es['log']['product_id'],
                            parent_entity=pd_es["sessions"],
                            primitive=NumUnique)
        single[0]

    error_text = "Cannot get item from slice of multi output feature"
    with pytest.raises(ValueError, match=error_text):
        three_common[0][0]

    error_text = 'index is higher than the number of outputs'
    with pytest.raises(AssertionError, match=error_text):
        three_common[10]
