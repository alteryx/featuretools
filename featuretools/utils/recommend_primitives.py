from typing import List

import pandas as pd

from featuretools.computational_backends import calculate_feature_matrix
from featuretools.entityset import EntitySet
from featuretools.primitives.utils import get_transform_primitives
from featuretools.synthesis import dfs, get_valid_primitives

ORDERED_PRIMITIVES = [  # primitives that require ordering
    "absolute_diff",
    "cum_sum",
    "cum_count",
    "cum_mean",
    "cum_max",
    "cum_min",
    "cumulative_time_since_last_false",
    "cumulative_time_since_last_true",
    "diff",
    "diff_datetime",
    "exponential_weighted_average",
    "exponential_weighted_std",
    "exponential_weighted_variance",
    "greater_than_previous",
    "is_first_occurrence",
    "is_last_occurrence",
    "is_max_so_far",
    "is_min_so_far",
    "lag",
    "less_than_previous",
    "percent_change",
    "same_as_previous",
    "time_since_previous",
]


DEPRECATED_PRIMITIVES = [
    "multiply_boolean",  # functionality duplicated by 'and' primitive
    "numeric_lag",  # deperecated and replaced with `lag`
]

REQUIRED_INPUT_PRIMITIVES = [
    "count_string",
    "distance_to_holiday",
    "is_in_geobox",
    "score_percentile",
    "subtract_numeric_scalar",
    "scalar_subtract_numeric_feature",
    "not_equal_scalar",
    "multiply_numeric_scalar",
    "modulo_numeric_scalar",
    "divide_numeric_scalar",
    "add_numeric_scalar",
    "equal_scalar",
    "greater_than_equal_to_scalar",
    "less_than_equal_to_scalar",
    "divide_by_feature",
    "greater_than_scalar",
    "less_than_scalar",
    "modulo_by_feature",
    "time_since",
    "savgol_filter",
    "isin",
    "numeric_bin",
]

OTHER_PRIMITIVES_TO_EXCLUDE = [  # Excluding all multi-input primitives and numeric primitives that don't handle skew
    "absolute",
    "negate",
    "not",
    "and",
    "or",
    "is_zero",
    "is_null",
    "greater_than_equal_to",
    "equal",
    "greater_than",
    "is_whole_number",
    "not_equal",
    "multiply_numeric",
    "multiply_numeric_boolean",
    "add_numeric",
    "less_than_equal_to",
    "divide_numeric",
    "less_than",
    "subtract_numeric",
    "modulo_numeric",
    "cosine",
    "tangent",
    "sine",
]

DEFAULT_EXCLUDED_PRIMITIVES = (
    REQUIRED_INPUT_PRIMITIVES
    + DEPRECATED_PRIMITIVES
    + ORDERED_PRIMITIVES
    + OTHER_PRIMITIVES_TO_EXCLUDE
)

# TODO: Make this list more dynamic
TIME_SERIES_PRIMITIVES = [
    "expanding_count",
    "expanding_max",
    "expanding_mean",
    "expanding_min",
    "expanding_std",
    "expanding_trend",
    "lag",
    "rolling_count",
    "rolling_outlier_count",
    "rolling_max",
    "rolling_mean",
    "rolling_min",
    "rolling_std",
    "rolling_trend",
]


# TODO: Support multi-table
def get_recommended_primitives(
    entityset: EntitySet,
    target_dataframe_name: str,
    is_time_series: bool,
    excluded_primitives: List[str] = DEFAULT_EXCLUDED_PRIMITIVES,
) -> List[str]:
    """Get a list of recommended primitives given an entity set.

    Description:
        Given a single table entity set with a `target_dataframe_name`, `is_time_series` specified
        and a list of any primitives in `excluded_primitives` to not be included in the final recommendation list.

    Args:
        entityset (EntitySet): EntitySet that only contains one dataframe.
        target_dataframe_name (str): Name of target dataframe to access in `entityset`.
        is_time_series (bool): Whether or not time-series analysis will be performed. If set to `True`, `Lag` will always be recommended,
        as well as all Rolling primitives if numeric columns are present.
        excluded_primitives (List[str]): List of transform primitives to exclude from recommendations.

    Note:
        This function currently only works for single table and will only recommend transform primitives.
    """
    recommended_primitives = set()

    if not is_time_series:
        excluded_primitives += TIME_SERIES_PRIMITIVES

    if is_time_series:
        recommended_primitives.add("lag")

    all_trans_primitives = get_transform_primitives()
    selected_trans_primitives = [
        p for name, p in all_trans_primitives.items() if name not in excluded_primitives
    ]

    valid_primitive_names = [
        prim.name
        for prim in get_valid_primitives(
            entityset,
            target_dataframe_name,
            1,
            selected_trans_primitives,
        )[1]
    ]

    recommended_primitives.update(
        _recommend_non_numeric_primitives(
            entityset,
            target_dataframe_name,
            valid_primitive_names,
        ),
    )
    skew_numeric_primitives = set(["square_root", "natural_logarithm"])
    valid_skew_primtives = skew_numeric_primitives.intersection(valid_primitive_names)

    time_series_primitives = set(TIME_SERIES_PRIMITIVES)
    valid_time_series_primitives = time_series_primitives.intersection(
        valid_primitive_names,
    )

    numeric_tags_only_df = entityset[target_dataframe_name].ww.select("numeric")

    if valid_skew_primtives:
        recommended_primitives.update(
            _recommend_skew_numeric_primitives(
                numeric_tags_only_df,
                valid_skew_primtives,
            ),
        )

    if valid_time_series_primitives:
        recommended_primitives.update(valid_time_series_primitives)
    return list(recommended_primitives)


def _recommend_non_numeric_primitives(
    entityset: EntitySet,
    target_dataframe_name: str,
    valid_primitives: List[str],
) -> set:
    """Get a set of recommended non-numeric primitives given an entity set.

    Description:
        Given a single table entity set with a `target_dataframe_name` and an applicable list of `valid_primitives`,
        get a set of primitives which produce non-unique features.

    Args:
        entityset (EntitySet): EntitySet that only contains one dataframe.
        target_dataframe_name (str): Name of target dataframe to access in `entityset`.
        valid_primitives (List[str]): List of primitives to calculate and check output features.
    """

    recommended_non_numeric_primitives = set()
    # Only want to run feature generation on non numeric primitives
    numeric_columns_to_ignore = list(
        entityset[target_dataframe_name].ww.select(include="numeric").columns,
    )
    features = dfs(
        entityset=entityset,
        target_dataframe_name=target_dataframe_name,
        trans_primitives=valid_primitives,
        max_depth=1,
        features_only=True,
        ignore_columns={target_dataframe_name: numeric_columns_to_ignore},
    )

    for f in features:
        if (
            f.primitive.name is not None
            and f.primitive.name not in recommended_non_numeric_primitives
        ):
            try:
                matrix = calculate_feature_matrix([f], entityset)
                for f_name in f.get_feature_names():
                    if len(matrix[f_name].unique()) > 1:
                        recommended_non_numeric_primitives.add(f.primitive.name)
            except Exception:  # If error in calculating feature matrix pass on the recommendation
                pass

    return recommended_non_numeric_primitives


def _recommend_skew_numeric_primitives(
    numerics_only_df: pd.DataFrame,
    valid_skew_primtives: set,
) -> set:
    """Get a set of recommended skew numeric primitives given an entity set.

    Description:
        Given woodwork initialized dataframe of origin features with only `numeric` semantic tags and an applicable list of `valid_skew_primitives`,
        get a set of primitives which could be applied to address right skewness.

    Args:
        numerics_only_df (pd.DataFrame): Woodwork initialized dataframe of origin features with only `numeric` semantic tags
        valid_skew_primitives (set): set of valid skew primitives (square_root or natural_logarithm) to potentially recommend.

    Note:
        We currently only have primitives to address right skewness.
    """
    recommended_skew_primitives = set()
    for col in numerics_only_df:
        # Shouldn't recommend log, sqrt if nans, zeros and negative numbers are present
        contains_nan = numerics_only_df[col].isnull().values.any()
        all_above_zero = (numerics_only_df[col] > 0).all()
        if all_above_zero and not contains_nan:
            skew = numerics_only_df[col].skew()
            # We currently don't have anything in featuretools to automatically handle left skewed data as well as skewed data with negative values
            if skew > 0.5 and skew < 1 and "square_root" in valid_skew_primtives:
                recommended_skew_primitives.add("square_root")
                # TODO: Add Box Cox here when available
            if skew > 1 and "natural_logarithm" in valid_skew_primtives:
                recommended_skew_primitives.add("natural_logarithm")
                # TODO: Add log base 10 transform primitive when available
    return recommended_skew_primitives
