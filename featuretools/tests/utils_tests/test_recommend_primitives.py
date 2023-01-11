import pandas as pd
import pytest

from featuretools import EntitySet
from featuretools.utils.recommend_primitives import (
    DEFAULT_EXCLUDED_PRIMITIVES,
    TIME_SERIES_PRIMITIVES,
    _recommend_non_numeric_primitives,
    _recommend_skew_numeric_primitives,
    get_recommended_primitives,
)


@pytest.fixture
def moderate_right_skewed_df():
    return pd.DataFrame(
        {"moderately right skewed": [2, 3, 4, 4, 4, 5, 5, 7, 9, 11, 12, 13, 15]},
    )


@pytest.fixture
def heavy_right_skewed_df():
    return pd.DataFrame(
        {"heavy right skewed": [1, 1, 1, 1, 2, 2, 3, 3, 4, 5, 9, 11, 13]},
    )


@pytest.fixture
def left_skewed_df():
    return pd.DataFrame(
        {"left skewed": [2, 3, 4, 5, 7, 9, 11, 11, 11, 12, 12, 12, 13, 15]},
    )


@pytest.fixture
def skewed_df_zeros():
    return pd.DataFrame({"zeros": [-1, 0, 0, 1, 2, 2, 3, 4, 5, 7, 9]})


@pytest.fixture
def normal_df():
    return pd.DataFrame({"normal": [2, 3, 4, 5, 5, 6, 6, 7, 7, 8, 9, 10, 11]})


@pytest.fixture
def right_skew_moderate_and_heavy_df(moderate_right_skewed_df, heavy_right_skewed_df):
    return pd.concat([moderate_right_skewed_df, heavy_right_skewed_df], axis=1)


def test_recommend_skew_numeric_primitives(
    moderate_right_skewed_df,
    heavy_right_skewed_df,
    left_skewed_df,
    skewed_df_zeros,
    normal_df,
    right_skew_moderate_and_heavy_df,
):
    valid_skew_primtives = set(["square_root", "natural_logarithm"])

    assert _recommend_skew_numeric_primitives(
        moderate_right_skewed_df,
        valid_skew_primtives,
    ) == set(["square_root"])
    assert _recommend_skew_numeric_primitives(
        heavy_right_skewed_df,
        valid_skew_primtives,
    ) == set(["natural_logarithm"])
    assert (
        _recommend_skew_numeric_primitives(left_skewed_df, valid_skew_primtives)
        == set()
    )
    assert (
        _recommend_skew_numeric_primitives(skewed_df_zeros, valid_skew_primtives)
        == set()
    )
    assert _recommend_skew_numeric_primitives(normal_df, valid_skew_primtives) == set()
    assert (
        _recommend_skew_numeric_primitives(
            right_skew_moderate_and_heavy_df,
            valid_skew_primtives,
        )
        == valid_skew_primtives
    )


def test_recommend_non_numeric_primitives(make_es):
    ecom_es_customers = EntitySet("ES without numerics")
    ecom_es_customers.add_dataframe(make_es["customers"])
    valid_primitives = [
        "day",
        "num_characters",
        "natural_logarithm",
        "sine",
    ]
    actual_recomendations = _recommend_non_numeric_primitives(
        ecom_es_customers,
        "customers",
        valid_primitives,
    )
    expected_recomendations = set(
        [
            "day",
            "num_characters",
        ],
    )
    assert expected_recomendations == actual_recomendations


def test_get_recommended_primitives(make_es):
    ecom_es_customers = EntitySet()
    ecom_es_customers.add_dataframe(make_es["customers"])
    prims_to_exclude = DEFAULT_EXCLUDED_PRIMITIVES + ["latlong_to_city"]
    actual_recomendations = get_recommended_primitives(
        ecom_es_customers,
        False,
        prims_to_exclude,
    )
    expected_recomendations = [
        "day",
        "num_characters",
        "natural_logarithm",
        "punctuation_count",
        "mean_characters_per_word",
        "is_weekend",
        "whitespace_count",
        "median_word_length",
        "month",
        "total_word_length",
        "weekday",
        "day_of_year",
        "week",
        "quarter",
        "email_address_to_domain",
        "number_of_common_words",
        "num_words",
        "num_unique_separators",
        "age",
        "year",
        "is_leap_year",
        "days_in_month",
        "is_free_email_domain",
        "number_of_unique_words",
    ]
    for prim in expected_recomendations:
        assert prim in actual_recomendations

    for ts_prim in TIME_SERIES_PRIMITIVES:
        assert ts_prim not in actual_recomendations


def test_get_recommended_primitives_time_series(make_es):
    ecom_es_log = EntitySet()
    ecom_es_log.add_dataframe(make_es["log"])
    prims_to_exclude = DEFAULT_EXCLUDED_PRIMITIVES + ["latlong_to_city"]
    actual_recomendations_ts = get_recommended_primitives(
        ecom_es_log,
        True,
        prims_to_exclude,
    )
    for ts_prim in TIME_SERIES_PRIMITIVES:
        assert ts_prim in actual_recomendations_ts


def test_get_recommended_primtives_exclude(make_es):
    ecom_es_customers = EntitySet()
    ecom_es_customers.add_dataframe(make_es["customers"])
    extra_exclude = ["num_characters", "natural_logarithm"]
    prims_to_exclude = DEFAULT_EXCLUDED_PRIMITIVES + ["latlong_to_city"] + extra_exclude
    actual_recomendations = get_recommended_primitives(
        ecom_es_customers,
        False,
        prims_to_exclude,
    )

    for ex_prim in extra_exclude:
        assert ex_prim not in actual_recomendations


def test_get_recommended_primitives_empty_es_error():
    error_msg = "No DataFrame in EntitySet found. Please add a DataFrame."
    empty_es = EntitySet()
    with pytest.raises(IndexError, match=error_msg):
        get_recommended_primitives(
            empty_es,
            False,
        )


def test_get_recommended_primitives_multi_table_es_error(make_es):
    error_msg = "Multi-table EntitySets are currently not supported. Please only use a single table EntitySet."
    with pytest.raises(IndexError, match=error_msg):
        get_recommended_primitives(
            make_es,
            False,
        )
