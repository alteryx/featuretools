import os.path

import pytest
from pympler.asizeof import asizeof

import featuretools as ft
from featuretools import config
from featuretools.feature_base import IdentityFeature
from featuretools.primitives import (
    Count,
    Diff,
    Last,
    Mode,
    Negate,
    NMostCommon,
    NumUnique,
    Sum,
    TransformPrimitive
)
from featuretools.tests.testing_utils import check_rename
from featuretools.variable_types import Categorical, Datetime, Id, Numeric


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
    ignored = set([agg1.unique_name()])
    deep_ignored = d1.get_dependencies(deep=True, ignored=ignored)
    assert [s.unique_name() for s in shallow] == [agg2.unique_name()]
    assert [d.unique_name() for d in deep] == [agg2.unique_name(), agg1.unique_name(), f.unique_name()]
    assert [d.unique_name() for d in deep_ignored] == [agg2.unique_name()]


def test_get_depth(es):
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
    assert squared.base_features[0].unique_name() == squared.base_features[1].unique_name()


def test_return_type_inference(es):
    mode = ft.Feature(es["log"]["priority_level"], parent_entity=es["customers"], primitive=Mode)
    assert mode.variable_type == es["log"]["priority_level"].__class__


def test_return_type_inference_direct_feature(es):
    mode = ft.Feature(es["log"]["priority_level"], parent_entity=es["customers"], primitive=Mode)
    mode_session = ft.Feature(mode, es["sessions"])
    assert mode_session.variable_type == es["log"]["priority_level"].__class__


def test_return_type_inference_index(es):
    last = ft.Feature(es["log"]["id"], parent_entity=es["customers"], primitive=Last)
    assert last.variable_type == Categorical


def test_return_type_inference_datetime_time_index(es):
    last = ft.Feature(es["log"]["datetime"], parent_entity=es["customers"], primitive=Last)
    assert last.variable_type == Datetime


def test_return_type_inference_numeric_time_index(int_es):
    last = ft.Feature(int_es["log"]["datetime"], parent_entity=int_es["customers"], primitive=Last)
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


def test_to_dictionary_direct(es):
    direct_feature = ft.Feature(es["sessions"]["customer_id"], es["log"])

    expected = {
        'type': 'DirectFeature',
        'dependencies': ['sessions: customer_id'],
        'arguments': {'name': None,
                      'base_feature': 'sessions: customer_id',
                      'relationship': {'parent_entity_id': 'sessions',
                                       'child_entity_id': 'log',
                                       'parent_variable_id': 'id',
                                       'child_variable_id': 'session_id'}
                      }
    }

    assert expected == direct_feature.to_dictionary()


def test_to_dictionary_identity(es):
    identity_feature = ft.Feature(es["sessions"]["customer_id"])

    expected = {
        'type': 'IdentityFeature',
        'dependencies': [],
        'arguments': {'name': None,
                      'variable_id': 'customer_id',
                      'entity_id': 'sessions'}
    }

    assert expected == identity_feature.to_dictionary()


def test_to_dictionary_agg(es):
    agg_feature = ft.Feature(es["customers"]["age"],
                             primitive=Sum, parent_entity=es['cohorts'])

    expected = {
        'type': 'AggregationFeature',
        'dependencies': ['customers: age'],
        'arguments': {'name': None,
                      'base_features': ['customers: age'],
                      'relationship_path': [{'parent_entity_id': 'cohorts',
                                             'child_entity_id': 'customers',
                                             'parent_variable_id': 'cohort',
                                             'child_variable_id': 'cohort'}],
                      'primitive': {'type': 'Sum',
                                    'module': 'featuretools.primitives.standard.aggregation_primitives',
                                    'arguments': {}},
                      'where': None,
                      'use_previous': None}
    }

    assert expected == agg_feature.to_dictionary()


def test_to_dictionary_where(es):
    agg_where = ft.Feature(es['log']['value'], parent_entity=es['sessions'],
                           where=ft.IdentityFeature(es['log']['value']) == 2, primitive=Sum)

    expected = {
        'type': 'AggregationFeature',
        'dependencies': ['log: value', 'log: value = 2'],
        'arguments': {'name': None,
                      'base_features': ['log: value'],
                      'relationship_path': [{'parent_entity_id': 'sessions',
                                             'child_entity_id': 'log',
                                             'parent_variable_id': 'id',
                                             'child_variable_id': 'session_id'}],
                      'primitive': {'type': 'Sum',
                                    'module': 'featuretools.primitives.standard.aggregation_primitives',
                                    'arguments': {}},
                      'where': 'log: value = 2',
                      'use_previous': None}
    }

    assert expected == agg_where.to_dictionary()


def test_to_dictionary_trans(es):
    trans_feature = ft.Feature(es["customers"]["age"], primitive=Negate)

    expected = {
        'type': 'TransformFeature',
        'dependencies': ['customers: age'],
        'arguments': {'name': None,
                      'base_features': ['customers: age'],
                      'primitive': {'type': 'Negate',
                                    'module': 'featuretools.primitives.standard.transform_primitive',
                                    'arguments': {}}
                      }}

    assert expected == trans_feature.to_dictionary()


def test_to_dictionary_grouby_trans(es):
    groupby_feature = ft.Feature(es['log']['value'], primitive=Negate, groupby=es['log']['product_id'])

    expected = {
        'type': 'GroupByTransformFeature',
        'dependencies': ['log: value', 'log: product_id'],
        'arguments': {'name': None,
                      'base_features': ['log: value'],
                      'primitive': {'type': 'Negate',
                                    'module': 'featuretools.primitives.standard.transform_primitive',
                                    'arguments': {}},
                      'groupby': 'log: product_id'}
    }

    assert expected == groupby_feature.to_dictionary()


def test_to_dictionary_multi_slice(es):
    slice_feature = ft.Feature(es['log']['product_id'], parent_entity=es['customers'], primitive=NMostCommon(n=2))[0]

    expected = {
        'type': 'FeatureOutputSlice',
        'dependencies': ['customers: N_MOST_COMMON(log.product_id, n=2)'],
        'arguments': {'name': None,
                      'base_feature': 'customers: N_MOST_COMMON(log.product_id, n=2)',
                      'n': 0
                      }}

    assert expected == slice_feature.to_dictionary()


def test_multi_output_base_error_agg(es):
    three_common = NMostCommon(3)
    tc = ft.Feature(es['log']['product_id'], parent_entity=es["sessions"], primitive=three_common)
    error_text = "Cannot stack on whole multi-output feature."
    with pytest.raises(ValueError, match=error_text):
        ft.Feature(tc, parent_entity=es['customers'], primitive=NumUnique)


def test_multi_output_base_error_trans(es):
    class TestTime(TransformPrimitive):
        name = "test_time"
        input_types = [Datetime]
        return_type = Numeric
        number_output_features = 6

    tc = ft.Feature(es['customers']['date_of_birth'], primitive=TestTime)

    error_text = "Cannot stack on whole multi-output feature."
    with pytest.raises(ValueError, match=error_text):
        ft.Feature(tc, primitive=Diff)


def test_multi_output_attributes(es):
    tc = ft.Feature(es['log']['product_id'], parent_entity=es["sessions"], primitive=NMostCommon)

    assert tc.generate_name() == 'N_MOST_COMMON(log.product_id)'
    assert tc.number_output_features == 3
    assert tc.base_features == ['<Feature: product_id>']

    assert tc[0].generate_name() == 'N_MOST_COMMON(log.product_id)[0]'
    assert tc[0].number_output_features == 1
    assert tc[0].base_features == [tc]
    assert tc.relationship_path == tc[0].relationship_path


def test_multi_output_index_error(es):
    error_text = "can only access slice of multi-output feature"
    three_common = ft.Feature(es['log']['product_id'],
                              parent_entity=es["sessions"],
                              primitive=NMostCommon)

    with pytest.raises(AssertionError, match=error_text):
        single = ft.Feature(es['log']['product_id'],
                            parent_entity=es["sessions"],
                            primitive=NumUnique)
        single[0]

    error_text = "Cannot get item from slice of multi output feature"
    with pytest.raises(ValueError, match=error_text):
        three_common[0][0]

    error_text = 'index is higher than the number of outputs'
    with pytest.raises(AssertionError, match=error_text):
        three_common[10]


def test_rename(es):
    feat = ft.Feature(es['log']['id'], parent_entity=es['sessions'], primitive=Count)
    new_name = 'session_test'
    new_names = ['session_test']
    check_rename(feat, new_name, new_names)


def test_rename_multioutput(es):
    feat = ft.Feature(es['log']['product_id'],
                      parent_entity=es['customers'],
                      primitive=NMostCommon(n=2))
    new_name = 'session_test'
    new_names = ['session_test[0]', 'session_test[1]']
    check_rename(feat, new_name, new_names)


def test_rename_featureoutputslice(es):
    multi_output_feat = ft.Feature(es['log']['product_id'],
                                   parent_entity=es['customers'],
                                   primitive=NMostCommon(n=2))
    feat = ft.feature_base.FeatureOutputSlice(multi_output_feat, 0)
    new_name = 'session_test'
    new_names = ['session_test']
    check_rename(feat, new_name, new_names)
