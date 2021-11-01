import os.path

import pytest
from pympler.asizeof import asizeof
from woodwork.column_schema import ColumnSchema
from woodwork.logical_types import Datetime, Integer

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


def test_copy_features_does_not_copy_entityset(es):
    agg = ft.Feature(es['log'].ww['value'], parent_dataframe_name='sessions', primitive=Sum)
    agg_where = ft.Feature(es['log'].ww['value'], parent_dataframe_name='sessions',
                           where=IdentityFeature(es['log'].ww['value']) == 2, primitive=Sum)
    agg_use_previous = ft.Feature(es['log'].ww['value'], parent_dataframe_name='sessions',
                                  use_previous='4 days', primitive=Sum)
    agg_use_previous_where = ft.Feature(es['log'].ww['value'], parent_dataframe_name='sessions',
                                        where=IdentityFeature(es['log'].ww['value']) == 2,
                                        use_previous='4 days', primitive=Sum)
    features = [agg, agg_where, agg_use_previous, agg_use_previous_where]
    in_memory_size = asizeof(locals())
    copied = [f.copy() for f in features]
    new_in_memory_size = asizeof(locals())
    assert new_in_memory_size < 2 * in_memory_size


def test_get_dependencies(es):
    f = ft.Feature(es['log'].ww['value'])
    agg1 = ft.Feature(f, parent_dataframe_name='sessions', primitive=Sum)
    agg2 = ft.Feature(agg1, parent_dataframe_name='customers', primitive=Sum)
    d1 = ft.Feature(agg2, 'sessions')
    shallow = d1.get_dependencies(deep=False, ignored=None)
    deep = d1.get_dependencies(deep=True, ignored=None)
    ignored = set([agg1.unique_name()])
    deep_ignored = d1.get_dependencies(deep=True, ignored=ignored)
    assert [s.unique_name() for s in shallow] == [agg2.unique_name()]
    assert [d.unique_name() for d in deep] == [agg2.unique_name(), agg1.unique_name(), f.unique_name()]
    assert [d.unique_name() for d in deep_ignored] == [agg2.unique_name()]


def test_get_depth(es):
    f = ft.Feature(es['log'].ww['value'])
    g = ft.Feature(es['log'].ww['value'])
    agg1 = ft.Feature(f, parent_dataframe_name='sessions', primitive=Last)
    agg2 = ft.Feature(agg1, parent_dataframe_name='customers', primitive=Last)
    d1 = ft.Feature(agg2, 'sessions')
    d2 = ft.Feature(d1, 'log')
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
    feature = ft.Feature(es['log'].ww['value'])
    squared = feature * feature
    assert len(squared.base_features) == 2
    assert squared.base_features[0].unique_name() == squared.base_features[1].unique_name()


def test_return_type_inference(es):
    mode = ft.Feature(es["log"].ww["priority_level"], parent_dataframe_name="customers", primitive=Mode)
    assert mode.column_schema == IdentityFeature(es["log"].ww["priority_level"]).column_schema


def test_return_type_inference_direct_feature(es):
    mode = ft.Feature(es["log"].ww["priority_level"], parent_dataframe_name="customers", primitive=Mode)
    mode_session = ft.Feature(mode, "sessions")
    assert mode_session.column_schema == IdentityFeature(es["log"].ww["priority_level"]).column_schema


def test_return_type_inference_index(es):
    last = ft.Feature(es["log"].ww["id"], parent_dataframe_name="customers", primitive=Last)
    assert "index" not in last.column_schema.semantic_tags
    assert isinstance(last.column_schema.logical_type, Integer)


def test_return_type_inference_datetime_time_index(es):
    last = ft.Feature(es["log"].ww["datetime"], parent_dataframe_name="customers", primitive=Last)
    assert isinstance(last.column_schema.logical_type, Datetime)


def test_return_type_inference_numeric_time_index(int_es):
    last = ft.Feature(int_es["log"].ww["datetime"], parent_dataframe_name="customers", primitive=Last)
    assert "numeric" in last.column_schema.semantic_tags


def test_return_type_inference_id(es):
    # direct features should keep foreign key tag
    direct_id_feature = ft.Feature(es["sessions"].ww["customer_id"], "log")
    assert "foreign_key" in direct_id_feature.column_schema.semantic_tags

    # aggregations of foreign key types should get converted
    last_feat = ft.Feature(es["log"].ww["session_id"], parent_dataframe_name="customers", primitive=Last)
    assert "foreign_key" not in last_feat.column_schema.semantic_tags
    assert isinstance(last_feat.column_schema.logical_type, Integer)

    # also test direct feature of aggregation
    last_direct = ft.Feature(last_feat, "sessions")
    assert "foreign_key" not in last_direct.column_schema.semantic_tags
    assert isinstance(last_direct.column_schema.logical_type, Integer)


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
    actual = ft.Feature(IdentityFeature(es["sessions"].ww["customer_id"]), "log").to_dictionary()

    expected = {
        'type': 'DirectFeature',
        'dependencies': ['sessions: customer_id'],
        'arguments': {'name': None,
                      'base_feature': 'sessions: customer_id',
                      'relationship': {'parent_dataframe_name': 'sessions',
                                       'child_dataframe_name': 'log',
                                       'parent_column_name': 'id',
                                       'child_column_name': 'session_id'}
                      }
    }

    assert expected == actual


def test_to_dictionary_identity(es):
    actual = ft.Feature(es["sessions"].ww["customer_id"]).to_dictionary()

    expected = {
        'type': 'IdentityFeature',
        'dependencies': [],
        'arguments': {'name': None,
                      'column_name': 'customer_id',
                      'dataframe_name': 'sessions'}
    }

    assert expected == actual


def test_to_dictionary_agg(es):
    actual = ft.Feature(es["customers"].ww["age"], primitive=Sum, parent_dataframe_name='cohorts').to_dictionary()

    expected = {
        'type': 'AggregationFeature',
        'dependencies': ['customers: age'],
        'arguments': {'name': None,
                      'base_features': ['customers: age'],
                      'relationship_path': [{'parent_dataframe_name': 'cohorts',
                                             'child_dataframe_name': 'customers',
                                             'parent_column_name': 'cohort',
                                             'child_column_name': 'cohort'}],
                      'primitive': {'type': 'Sum',
                                    'module': 'featuretools.primitives.standard.aggregation_primitives',
                                    'arguments': {}},
                      'where': None,
                      'use_previous': None}
    }

    assert expected == actual


def test_to_dictionary_where(es):
    actual = ft.Feature(es['log'].ww['value'], parent_dataframe_name='sessions',
                        where=ft.IdentityFeature(es['log'].ww['value']) == 2, primitive=Sum).to_dictionary()

    expected = {
        'type': 'AggregationFeature',
        'dependencies': ['log: value', 'log: value = 2'],
        'arguments': {'name': None,
                      'base_features': ['log: value'],
                      'relationship_path': [{'parent_dataframe_name': 'sessions',
                                             'child_dataframe_name': 'log',
                                             'parent_column_name': 'id',
                                             'child_column_name': 'session_id'}],
                      'primitive': {'type': 'Sum',
                                    'module': 'featuretools.primitives.standard.aggregation_primitives',
                                    'arguments': {}},
                      'where': 'log: value = 2',
                      'use_previous': None}
    }

    assert expected == actual


def test_to_dictionary_trans(es):
    trans_feature = ft.Feature(es["customers"].ww["age"], primitive=Negate)

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


def test_to_dictionary_groupby_trans(es):
    id_feat = ft.Feature(es['log'].ww['product_id'])
    groupby_feature = ft.Feature(es['log'].ww['value'], primitive=Negate, groupby=id_feat)

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
    slice_feature = ft.Feature(es['log'].ww['product_id'], parent_dataframe_name='customers', primitive=NMostCommon(n=2))[0]

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
    tc = ft.Feature(es['log'].ww['product_id'], parent_dataframe_name="sessions", primitive=three_common)
    error_text = "Cannot stack on whole multi-output feature."
    with pytest.raises(ValueError, match=error_text):
        ft.Feature(tc, parent_dataframe_name='customers', primitive=NumUnique)


def test_multi_output_base_error_trans(es):
    class TestTime(TransformPrimitive):
        name = "test_time"
        input_types = [ColumnSchema(logical_type=Datetime)]
        return_type = ColumnSchema(semantic_tags={'numeric'})
        number_output_features = 6

    tc = ft.Feature(es['customers'].ww['birthday'], primitive=TestTime)

    error_text = "Cannot stack on whole multi-output feature."
    with pytest.raises(ValueError, match=error_text):
        ft.Feature(tc, primitive=Diff)


def test_multi_output_attributes(es):
    tc = ft.Feature(es['log'].ww['product_id'], parent_dataframe_name="sessions", primitive=NMostCommon)

    assert tc.generate_name() == 'N_MOST_COMMON(log.product_id)'
    assert tc.number_output_features == 3
    assert tc.base_features == ['<Feature: product_id>']

    assert tc[0].generate_name() == 'N_MOST_COMMON(log.product_id)[0]'
    assert tc[0].number_output_features == 1
    assert tc[0].base_features == [tc]
    assert tc.relationship_path == tc[0].relationship_path


def test_multi_output_index_error(es):
    error_text = "can only access slice of multi-output feature"
    three_common = ft.Feature(es['log'].ww['product_id'],
                              parent_dataframe_name="sessions",
                              primitive=NMostCommon)

    with pytest.raises(AssertionError, match=error_text):
        single = ft.Feature(es['log'].ww['product_id'],
                            parent_dataframe_name="sessions",
                            primitive=NumUnique)
        single[0]

    error_text = "Cannot get item from slice of multi output feature"
    with pytest.raises(ValueError, match=error_text):
        three_common[0][0]

    error_text = 'index is higher than the number of outputs'
    with pytest.raises(AssertionError, match=error_text):
        three_common[10]


def test_rename(es):
    feat = ft.Feature(es['log'].ww['id'], parent_dataframe_name='sessions', primitive=Count)
    new_name = 'session_test'
    new_names = ['session_test']
    check_rename(feat, new_name, new_names)


def test_rename_multioutput(es):
    feat = ft.Feature(es['log'].ww['product_id'],
                      parent_dataframe_name='customers',
                      primitive=NMostCommon(n=2))
    new_name = 'session_test'
    new_names = ['session_test[0]', 'session_test[1]']
    check_rename(feat, new_name, new_names)


def test_rename_featureoutputslice(es):
    multi_output_feat = ft.Feature(es['log'].ww['product_id'],
                                   parent_dataframe_name='customers',
                                   primitive=NMostCommon(n=2))
    feat = ft.feature_base.FeatureOutputSlice(multi_output_feat, 0)
    new_name = 'session_test'
    new_names = ['session_test']
    check_rename(feat, new_name, new_names)
