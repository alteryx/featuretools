import logging
from typing import List

from featuretools.computational_backends import calculate_feature_matrix
from featuretools.entityset import EntitySet
from featuretools.primitives.utils import get_transform_primitives
from featuretools.synthesis import dfs, get_valid_primitives

ORDERED_PRIMITIVES = [  # non-numeric primitives that require specific ordering or a time index to be set
    "cum_count",
    "cumulative_time_since_last_false",
    "cumulative_time_since_last_true",
    "diff",
    "diff_datetime",
    "is_first_occurrence",
    "is_last_occurrence",
    "time_since_previous",
]


DEPRECATED_PRIMITIVES = [
    "multiply_boolean",  # functionality duplicated by 'and' primitive
    "numeric_lag",  # deprecated and replaced with `lag`
]

REQUIRED_INPUT_PRIMITIVES = [  # non-numeric primitives that require input
    "count_string",
    "distance_to_holiday",
    "is_in_geobox",
    "not_equal_scalar",
    "equal_scalar",
    "time_since",
    "isin",
]

OTHER_PRIMITIVES_TO_EXCLUDE = [  # Excluding some primitives that can produce too many features or aren't useful in extracting information
    "not",
    "and",
    "or",
    "equal",
    "not_equal",
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
    include_time_series_primitives: bool = False,
    excluded_primitives: List[str] = DEFAULT_EXCLUDED_PRIMITIVES,
) -> List[str]:
    """Get a list of recommended primitives given an entity set.

    Description:
        This function works by first getting a list of valid primitives withholding any primitives specified in `excluded_primitives` that could be applied to a single-table EntitySet.
        Secondly, engineered features are created for non-numeric fields and are checked for non-uniqueness. If the feature is non-unique, it is added to the recommendation list.
        Then, numeric fields are checked for skewness. Depending on how skew a column is `square_root` or `natural_logarithm` will be recommended.
        Lastly if `include_time_series_primitives` is specified as `True`, `Lag` will always be recommended,
        as well as all Rolling and Expanding primitives if numeric columns are present.

    Args:
        entityset (EntitySet): EntitySet that only contains one dataframe.
        include_time_series_primitives (bool): Whether or not time-series primitives should be considered. Defaults to False.
        excluded_primitives (List[str]): List of transform primitives to exclude from recommendations. Defaults to DEFAULT_EXCLUDED_PRIMITIVES.

    Note:
        The main objective of this function is to recommend primitives that could potentially provide important features to the modeling process.
        Non-numeric primitives do a great job in mainly serving as a way to extract information from origin features that may essentially be meaningless by themselves (e.g., NaturalLanguage, Datetime, LatLong).
        That is why they are the main focus of this function. Numeric transform primitives are very case-by-case dependent and therefore it is hard to mathematically quantify which should be recommended.
        Therefore, only transform primitives that address skewed numeric columns are included, as this is a standard and quantifiable transformation step. The only exception to this rule being
        for time series problems. Because there are so few primitives that are only applicable for time series, all of them are included in the recommended primitives list.

    Note:
        This function currently only works for single table and will only recommend transform primitives.
    """
    es_dataframe_list = entityset.dataframes
    if len(es_dataframe_list) == 0:
        raise IndexError("No DataFrame in EntitySet found. Please add a DataFrame.")
    if len(es_dataframe_list) > 1:
        raise IndexError(
            "Multi-table EntitySets are currently not supported. Please only use a single table EntitySet.",
        )

    target_dataframe_name = es_dataframe_list[0].ww.name

    recommended_primitives = set()

    if not include_time_series_primitives:
        excluded_primitives += TIME_SERIES_PRIMITIVES

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

    recommended_primitives.update(
        _recommend_skew_numeric_primitives(
            entityset,
            target_dataframe_name,
            valid_primitive_names,
        ),
    )

    recommended_primitives.update(
        set(TIME_SERIES_PRIMITIVES).intersection(
            valid_primitive_names,
        ),
    )
    return list(recommended_primitives)


def _recommend_non_numeric_primitives(
    entityset: EntitySet,
    target_dataframe_name: str,
    valid_primitives: List[str],
) -> set:
    """Get a set of non-numeric primitives for a given dataset and a list of primitives.

    Description:
        Given a single table entity set with a `target_dataframe_name` and an applicable list of `valid_primitives`,
        get a set of primitives which produce non-unique features.

    Args:
        entityset (EntitySet): EntitySet that only contains one dataframe.
        target_dataframe_name (str): Name of target dataframe to access in `entityset`.
        valid_primitives (List[str]): List of primitives to calculate and check output features.
    """

    recommended_non_numeric_primitives: set[str] = set()
    # Only want to run feature generation on non numeric primitives
    numeric_columns_to_ignore = list(
        entityset[target_dataframe_name]
        .ww.select(include="numeric", return_schema=True)
        .columns,
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
            except (
                Exception
            ) as e:  # If error in calculating feature matrix pass on the recommendation
                logger = logging.getLogger("featuretools")
                logger.error(
                    f"Exception with feature {f.get_name()} with primitive {f.primitive.name}: {str(e)}",
                )

    return recommended_non_numeric_primitives


def _recommend_skew_numeric_primitives(
    entityset: EntitySet,
    target_dataframe_name: str,
    valid_primitives: List[str],
) -> set:
    """Get a set of recommended skew numeric primitives given an entity set.

    Description:
        Given woodwork initialized dataframe of origin features with only `numeric` semantic tags and an applicable list of `valid_skew_primitives`,
        get a set of primitives which could be applied to address right skewness.

    Args:
        entityset (EntitySet): EntitySet that only contains one dataframe.
        target_dataframe_name (str): Name of target dataframe to access in `entityset`.
        valid_primitives (List[str]): List of primitives to compare.

    Note:
        We currently only have primitives to address right skewness.
    """
    recommended_skew_primitives: set[str] = set()
    skew_numeric_primitives = set(["square_root", "natural_logarithm"])
    valid_skew_primitives = skew_numeric_primitives.intersection(valid_primitives)
    if valid_skew_primitives:
        numerics_only_df = entityset[target_dataframe_name].ww.select("numeric")
        recommended_skew_primitives: set[str] = set()
        for col in numerics_only_df:
            # Shouldn't recommend log, sqrt if nans, zeros and negative numbers are present
            contains_nan = numerics_only_df[col].isnull().any()
            all_above_zero = (numerics_only_df[col] > 0).all()
            if all_above_zero and not contains_nan:
                skew = numerics_only_df[col].skew()
                # We currently don't have anything in featuretools to automatically handle left skewed data as well as skewed data with negative values
                if skew > 0.5 and skew < 1 and "square_root" in valid_skew_primitives:
                    recommended_skew_primitives.add("square_root")
                    # TODO: Add Box Cox here when available
                if skew > 1 and "natural_logarithm" in valid_skew_primitives:
                    recommended_skew_primitives.add("natural_logarithm")
                    # TODO: Add log base 10 transform primitive when available
    return recommended_skew_primitives
