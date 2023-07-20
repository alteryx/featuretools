import numpy as np
import pandas as pd
import pytest
from woodwork.column_schema import ColumnSchema
from woodwork.logical_types import Boolean, BooleanNullable, NaturalLanguage

from featuretools import EntitySet, Feature, dfs
from featuretools.selection import (
    remove_highly_correlated_features,
    remove_highly_null_features,
    remove_low_information_features,
    remove_single_value_features,
)
from featuretools.tests.testing_utils import make_ecommerce_entityset


@pytest.fixture
def feature_matrix():
    feature_matrix = pd.DataFrame(
        {
            "test": [0, 1, 2],
            "no_null": [np.nan, 0, 0],
            "some_null": [np.nan, 0, 0],
            "all_null": [np.nan, np.nan, np.nan],
            "many_value": [1, 2, 3],
            "dup_value": [1, 1, 2],
            "one_value": [1, 1, 1],
        },
    )
    return feature_matrix


@pytest.fixture
def test_es(pd_es, feature_matrix):
    pd_es.add_dataframe(dataframe_name="test", dataframe=feature_matrix, index="test")
    return pd_es


# remove low information features not supported in Dask
def test_remove_low_information_feature_names(feature_matrix):
    feature_matrix = remove_low_information_features(feature_matrix)
    assert feature_matrix.shape == (3, 5)
    assert "one_value" not in feature_matrix.columns
    assert "all_null" not in feature_matrix.columns


# remove low information features not supported in Dask
def test_remove_low_information_features(test_es, feature_matrix):
    features = [Feature(test_es["test"].ww[col]) for col in test_es["test"].columns]
    feature_matrix, features = remove_low_information_features(feature_matrix, features)
    assert feature_matrix.shape == (3, 5)
    assert len(features) == 5
    for f in features:
        assert f.get_name() in feature_matrix.columns
    assert "one_value" not in feature_matrix.columns
    assert "all_null" not in feature_matrix.columns


def test_remove_highly_null_features():
    nulls_df = pd.DataFrame(
        {
            "id": [0, 1, 2, 3],
            "half_nulls": [None, None, 88, 99],
            "all_nulls": [None, None, None, None],
            "quarter": ["a", "b", None, "c"],
            "vals": [True, True, False, False],
        },
    )

    es = EntitySet("data", {"nulls": (nulls_df, "id")})
    es["nulls"].ww.set_types(
        logical_types={"all_nulls": "categorical", "quarter": "categorical"},
    )
    fm, features = dfs(
        entityset=es,
        target_dataframe_name="nulls",
        trans_primitives=["is_null"],
        max_depth=1,
    )

    with pytest.raises(
        ValueError,
        match="pct_null_threshold must be a float between 0 and 1, inclusive.",
    ):
        remove_highly_null_features(fm, pct_null_threshold=1.1)

    with pytest.raises(
        ValueError,
        match="pct_null_threshold must be a float between 0 and 1, inclusive.",
    ):
        remove_highly_null_features(fm, pct_null_threshold=-0.1)

    no_thresh = remove_highly_null_features(fm)
    no_thresh_cols = set(no_thresh.columns)
    diff = set(fm.columns) - no_thresh_cols
    assert len(diff) == 1
    assert "all_nulls" not in no_thresh_cols

    half = remove_highly_null_features(fm, pct_null_threshold=0.5)
    half_cols = set(half.columns)
    diff = set(fm.columns) - half_cols
    assert len(diff) == 2
    assert "all_nulls" not in half_cols
    assert "half_nulls" not in half_cols

    no_tolerance = remove_highly_null_features(fm, pct_null_threshold=0)
    no_tolerance_cols = set(no_tolerance.columns)
    diff = set(fm.columns) - no_tolerance_cols
    assert len(diff) == 3
    assert "all_nulls" not in no_tolerance_cols
    assert "half_nulls" not in no_tolerance_cols
    assert "quarter" not in no_tolerance_cols

    (
        with_features_param,
        with_features_param_features,
    ) = remove_highly_null_features(fm, features)
    assert len(with_features_param_features) == len(no_thresh.columns)
    for i in range(len(with_features_param_features)):
        assert with_features_param_features[i].get_name() == no_thresh.columns[i]
        assert with_features_param.columns[i] == no_thresh.columns[i]


def test_remove_single_value_features():
    same_vals_df = pd.DataFrame(
        {
            "id": [0, 1, 2, 3],
            "all_numeric": [88, 88, 88, 88],
            "with_nan": [1, 1, None, 1],
            "all_nulls": [None, None, None, None],
            "all_categorical": ["a", "a", "a", "a"],
            "all_bools": [True, True, True, True],
            "diff_vals": ["hi", "bye", "bye", "hi"],
        },
    )

    es = EntitySet("data", {"single_vals": (same_vals_df, "id")})
    es["single_vals"].ww.set_types(
        logical_types={
            "all_nulls": "categorical",
            "all_categorical": "categorical",
            "diff_vals": "categorical",
        },
    )
    fm, features = dfs(
        entityset=es,
        target_dataframe_name="single_vals",
        trans_primitives=["is_null"],
        max_depth=1,
    )

    no_params, no_params_features = remove_single_value_features(fm, features)
    no_params_cols = set(no_params.columns)
    assert len(no_params_features) == 2
    assert "IS_NULL(with_nan)" in no_params_cols
    assert "diff_vals" in no_params_cols

    nan_as_value, nan_as_value_features = remove_single_value_features(
        fm,
        features,
        count_nan_as_value=True,
    )
    nan_cols = set(nan_as_value.columns)
    assert len(nan_as_value_features) == 3
    assert "IS_NULL(with_nan)" in nan_cols
    assert "diff_vals" in nan_cols
    assert "with_nan" in nan_cols

    without_features_param = remove_single_value_features(fm)
    assert len(no_params.columns) == len(without_features_param.columns)
    for i in range(len(no_params.columns)):
        assert no_params.columns[i] == without_features_param.columns[i]
        assert no_params_features[i].get_name() == without_features_param.columns[i]


def test_remove_highly_correlated_features():
    correlated_df = pd.DataFrame(
        {
            "id": [0, 1, 2, 3],
            "diff_ints": [34, 11, 29, 91],
            "words": ["test", "this is a short sentence", "foo bar", "baz"],
            "corr_words": [4, 24, 7, 3],
            "corr_1": [99, 88, 77, 33],
            "corr_2": [99, 88, 77, 33],
        },
    )

    es = EntitySet(
        "data",
        {"correlated": (correlated_df, "id", None, {"words": NaturalLanguage})},
    )
    fm, _ = dfs(
        entityset=es,
        target_dataframe_name="correlated",
        trans_primitives=["num_characters"],
        max_depth=1,
    )

    with pytest.raises(
        ValueError,
        match="pct_corr_threshold must be a float between 0 and 1, inclusive.",
    ):
        remove_highly_correlated_features(fm, pct_corr_threshold=1.1)

    with pytest.raises(
        ValueError,
        match="pct_corr_threshold must be a float between 0 and 1, inclusive.",
    ):
        remove_highly_correlated_features(fm, pct_corr_threshold=-0.1)

    with pytest.raises(
        AssertionError,
        match="feature named not_a_feature is not in feature matrix",
    ):
        remove_highly_correlated_features(fm, features_to_check=["not_a_feature"])

    to_check = remove_highly_correlated_features(
        fm,
        features_to_check=["corr_words", "NUM_CHARACTERS(words)", "diff_ints"],
    )
    to_check_columns = set(to_check.columns)
    assert len(to_check_columns) == 4
    assert "NUM_CHARACTERS(words)" not in to_check_columns
    assert "corr_1" in to_check_columns
    assert "corr_2" in to_check_columns

    to_keep = remove_highly_correlated_features(
        fm,
        features_to_keep=["NUM_CHARACTERS(words)"],
    )
    to_keep_names = set(to_keep.columns)
    assert len(to_keep_names) == 4
    assert "corr_words" in to_keep_names
    assert "NUM_CHARACTERS(words)" in to_keep_names
    assert "corr_2" not in to_keep_names

    new_fm = remove_highly_correlated_features(fm)
    assert len(new_fm.columns) == 3
    assert "corr_2" not in new_fm.columns
    assert "NUM_CHARACTERS(words)" not in new_fm.columns

    diff_threshold = remove_highly_correlated_features(fm, pct_corr_threshold=0.8)
    diff_threshold_cols = diff_threshold.columns
    assert len(diff_threshold_cols) == 2
    assert "corr_words" in diff_threshold_cols
    assert "diff_ints" in diff_threshold_cols


def test_remove_highly_correlated_features_init_woodwork():
    correlated_df = pd.DataFrame(
        {
            "id": [0, 1, 2, 3],
            "diff_ints": [34, 11, 29, 91],
            "words": ["test", "this is a short sentence", "foo bar", "baz"],
            "corr_words": [4, 24, 7, 3],
            "corr_1": [99, 88, 77, 33],
            "corr_2": [99, 88, 77, 33],
        },
    )

    es = EntitySet(
        "data",
        {"correlated": (correlated_df, "id", None, {"words": NaturalLanguage})},
    )
    fm, _ = dfs(
        entityset=es,
        target_dataframe_name="correlated",
        trans_primitives=["num_characters"],
        max_depth=1,
    )

    no_ww_fm = fm.copy()
    ww_fm = fm.copy()
    ww_fm.ww.init()

    new_no_ww_fm = remove_highly_correlated_features(no_ww_fm)
    new_ww_fm = remove_highly_correlated_features(ww_fm)

    pd.testing.assert_frame_equal(new_no_ww_fm, new_ww_fm)


def test_multi_output_selection():
    df1 = pd.DataFrame({"id": [0, 1, 2, 3]})

    df2 = pd.DataFrame(
        {
            "index": [0, 1, 2, 3],
            "first_id": [0, 1, 1, 3],
            "all_nulls": [None, None, None, None],
            "quarter": ["a", "b", None, "c"],
        },
    )

    dataframes = {
        "first": (df1, "id"),
        "second": (df2, "index"),
    }

    relationships = [("first", "id", "second", "first_id")]
    es = EntitySet("data", dataframes, relationships=relationships)
    es["second"].ww.set_types(
        logical_types={"all_nulls": "categorical", "quarter": "categorical"},
    )

    fm, features = dfs(
        entityset=es,
        target_dataframe_name="first",
        trans_primitives=[],
        agg_primitives=["n_most_common"],
        max_depth=1,
    )

    multi_output, multi_output_features = remove_single_value_features(fm, features)
    assert multi_output.columns == ["N_MOST_COMMON(second.quarter)[0]"]
    assert len(multi_output_features) == 1
    assert multi_output_features[0].get_name() == multi_output.columns[0]

    es = make_ecommerce_entityset()
    fm, features = dfs(
        entityset=es,
        target_dataframe_name="régions",
        trans_primitives=[],
        agg_primitives=["n_most_common"],
        max_depth=2,
    )

    matrix_with_slices, unsliced_features = remove_highly_null_features(fm, features)
    assert len(matrix_with_slices.columns) == 18
    assert len(unsliced_features) == 14

    matrix_columns = set(matrix_with_slices.columns)
    for f in unsliced_features:
        for f_name in f.get_feature_names():
            assert f_name in matrix_columns


def test_remove_highly_correlated_features_on_boolean_cols():
    correlated_df = pd.DataFrame(
        {
            "id": [0, 1, 2, 3],
            "diff_ints": [34, 11, 29, 91],
            "corr_words": [4, 24, 7, 3],
            "bools": [True, True, False, True],
        },
    )

    es = EntitySet(
        "data",
        {"correlated": (correlated_df, "id", None, {"bools": Boolean})},
    )

    feature_matrix, features = dfs(
        entityset=es,
        target_dataframe_name="correlated",
        trans_primitives=["equal"],
        agg_primitives=[],
        max_depth=1,
        return_types=[
            ColumnSchema(logical_type=BooleanNullable),
            ColumnSchema(logical_type=Boolean),
        ],
    )
    # Confirm both boolean logical types are included so that we know we're checking the correct types
    assert {
        ltype.type_string for ltype in feature_matrix.ww.logical_types.values()
    } == {Boolean.type_string, BooleanNullable.type_string}

    to_keep = remove_highly_correlated_features(
        feature_matrix=feature_matrix,
        features=features,
        pct_corr_threshold=0.3,
    )
    assert len(to_keep[0].columns) < len(feature_matrix.columns)
