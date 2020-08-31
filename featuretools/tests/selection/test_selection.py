import numpy as np
import pandas as pd
import pytest

import featuretools as ft
from featuretools import Feature
from featuretools.variable_types.variable import Text


@pytest.fixture
def feature_matrix():
    feature_matrix = pd.DataFrame({'test': [0, 1, 2],
                                   'no_null': [np.nan, 0, 0],
                                   'some_null': [np.nan, 0, 0],
                                   'all_null': [np.nan, np.nan, np.nan],
                                   'many_value': [1, 2, 3],
                                   'dup_value': [1, 1, 2],
                                   'one_value': [1, 1, 1]})
    return feature_matrix


@pytest.fixture
def test_es(pd_es, feature_matrix):
    pd_es.entity_from_dataframe('test', feature_matrix, index='test')
    return pd_es


# remove low information features not supported in Dask
def test_remove_low_information_feature_names(feature_matrix):
    feature_matrix = ft.selection.remove_low_information_features(feature_matrix)
    assert feature_matrix.shape == (3, 5)
    assert 'one_value' not in feature_matrix.columns
    assert 'all_null' not in feature_matrix.columns


# remove low information features not supported in Dask
def test_remove_low_information_features(test_es, feature_matrix):
    features = [Feature(v) for v in test_es['test'].variables]
    feature_matrix, features = ft.selection.remove_low_information_features(feature_matrix,
                                                                            features)
    assert feature_matrix.shape == (3, 5)
    assert len(features) == 5
    for f in features:
        assert f.get_name() in feature_matrix.columns
    assert 'one_value' not in feature_matrix.columns
    assert 'all_null' not in feature_matrix.columns


def test_remove_highly_null_features():
    nulls_df = pd.DataFrame({'id': [0, 1, 2, 3], 'half_nulls': [None, None, 88, 99],
                             "all_nulls": [None, None, None, None], 'quarter': ['a', 'b', None, 'c'], 'vals': [True, True, False, False]})

    es = ft.EntitySet("data", {'nulls': (nulls_df, 'id')})
    fm, _ = ft.dfs(entityset=es,
                   target_entity="nulls",
                   trans_primitives=['is_null'],
                   max_depth=2)

    with pytest.raises(ValueError, match='pct_null_threshold must be a float between 0 and 1, inclusive.'):
        ft.selection.remove_highly_null_features(fm, pct_null_threshold=1.1)

    no_thresh = ft.selection.remove_highly_null_features(fm)
    no_thresh_cols = set(no_thresh.columns)
    diff = set(fm.columns) - no_thresh_cols
    assert len(diff) == 1
    assert 'all_nulls' not in no_thresh_cols

    half = ft.selection.remove_highly_null_features(fm, pct_null_threshold=.5)
    half_cols = set(half.columns)
    diff = set(fm.columns) - half_cols
    assert len(diff) == 2
    assert 'all_nulls' not in half_cols
    assert 'half_nulls' not in half_cols

    no_tolerance = ft.selection.remove_highly_null_features(fm, pct_null_threshold=0)
    no_tolerance_cols = set(no_tolerance.columns)
    diff = set(fm.columns) - no_tolerance_cols
    assert len(diff) == 3
    assert 'all_nulls' not in no_tolerance_cols
    assert 'half_nulls' not in no_tolerance_cols
    assert 'quarter' not in no_tolerance_cols


def test_remove_single_value_features():
    same_vals_df = pd.DataFrame({'id': [0, 1, 2, 3], 'all_numeric': [88, 88, 88, 88], 'with_nan': [1, 1, None, 1],
                                 "all_nulls": [None, None, None, None], 'all_categorical': ['a', 'a', 'a', 'a'], 'all_bools': [True, True, True, True], 'diff_vals': ['hi', 'bye', 'bye', 'hi']})

    es = ft.EntitySet("data", {'single_vals': (same_vals_df, 'id')})
    fm, features = ft.dfs(entityset=es,
                          target_entity="single_vals",
                          trans_primitives=['is_null'],
                          max_depth=2)

    no_params, no_params_features = ft.selection.remove_single_value_features(fm, features)
    no_params_cols = set(no_params.columns)
    assert len(no_params_features) == 2
    assert 'IS_NULL(with_nan)' in no_params_cols
    assert 'diff_vals' in no_params_cols

    nan_as_value, nan_as_value_features = ft.selection.remove_single_value_features(fm, features, count_nan_as_value=True)
    nan_cols = set(nan_as_value.columns)
    assert len(nan_as_value_features) == 3
    assert 'IS_NULL(with_nan)' in nan_cols
    assert 'diff_vals' in nan_cols
    assert 'with_nan' in nan_cols


def test_remove_highly_correlated_features():
    correlated_df = pd.DataFrame({
        "id": [0, 1, 2, 3],
        "diff_ints": [34, 11, 29, 91],
        "words": ["test", "this is a short sentence", "foo bar", "baz"],
        "corr_words": [4, 24, 7, 3],
        'corr_1': [99, 88, 77, 33],
        'corr_2': [99, 88, 77, 33]
    })

    es = ft.EntitySet("data", {'correlated': (correlated_df, 'id', None, {'words': Text})})
    fm, _ = ft.dfs(entityset=es,
                   target_entity="correlated",
                   trans_primitives=['num_characters'],
                   max_depth=2)

    with pytest.raises(ValueError, match='pct_corr_threshold must be a float between 0 and 1, inclusive.'):
        ft.selection.remove_highly_correlated_features(fm, pct_corr_threshold=1.1)

    with pytest.raises(AssertionError, match="feature named not_a_feature is not in feature matrix"):
        ft.selection.remove_highly_correlated_features(fm, features_to_check=['not_a_feature'])

    to_check = ft.selection.remove_highly_correlated_features(fm, features_to_check=['corr_words', 'NUM_CHARACTERS(words)', 'diff_ints'])
    to_check_columns = set(to_check.columns)
    assert len(to_check_columns) == 3
    assert 'corr_words' not in to_check_columns
    assert 'NUM_CHARACTERS(words)' not in to_check_columns
    assert 'diff_ints' in to_check_columns

    to_keep = ft.selection.remove_highly_correlated_features(fm, features_to_keep=['NUM_CHARACTERS(words)'])
    to_keep_names = set(to_keep.columns)
    assert len(to_keep_names) == 2
    assert 'corr_words' not in to_keep_names
    assert 'NUM_CHARACTERS(words)' in to_keep_names

    new_fm = ft.selection.remove_highly_correlated_features(fm)
    assert len(new_fm.columns) == 1
    assert 'diff_ints' in new_fm.columns

    diff_threshold = ft.selection.remove_highly_correlated_features(fm, pct_corr_threshold=0.8)
    # --> this doesn't feel expected
    assert diff_threshold.columns == ['corr_2']
