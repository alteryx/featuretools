import pandas as pd
from woodwork.logical_types import Boolean, BooleanNullable


def remove_low_information_features(feature_matrix, features=None):
    """Select features that have at least 2 unique values and that are not all null

    Args:
        feature_matrix (:class:`pd.DataFrame`): DataFrame whose columns are feature names and rows are instances
        features (list[:class:`featuretools.FeatureBase`] or list[str], optional): List of features to select

    Returns:
        (feature_matrix, features)

    """
    keep = [
        c
        for c in feature_matrix
        if (
            feature_matrix[c].nunique(dropna=False) > 1
            and feature_matrix[c].dropna().shape[0] > 0
        )
    ]
    feature_matrix = feature_matrix[keep]
    if features is not None:
        features = [f for f in features if f.get_name() in feature_matrix.columns]
        return feature_matrix, features
    return feature_matrix


def remove_highly_null_features(feature_matrix, features=None, pct_null_threshold=0.95):
    """
    Removes columns from a feature matrix that have higher than a set threshold
    of null values.

    Args:
        feature_matrix (:class:`pd.DataFrame`): DataFrame whose columns are feature names and rows are instances.
        features (list[:class:`featuretools.FeatureBase`] or list[str], optional): List of features to select.
        pct_null_threshold (float): If the percentage of NaN values in an input feature exceeds this amount,
                that feature will be considered highly-null. Defaults to 0.95.

    Returns:
        pd.DataFrame, list[:class:`.FeatureBase`]:
            The feature matrix and the list of generated feature definitions. Matches dfs output.
            If no feature list is provided as input, the feature list will not be returned.
    """
    if pct_null_threshold < 0 or pct_null_threshold > 1:
        raise ValueError(
            "pct_null_threshold must be a float between 0 and 1, inclusive.",
        )

    percent_null_by_col = (feature_matrix.isnull().mean()).to_dict()

    if pct_null_threshold == 0.0:
        keep = [
            f_name
            for f_name, pct_null in percent_null_by_col.items()
            if pct_null <= pct_null_threshold
        ]
    else:
        keep = [
            f_name
            for f_name, pct_null in percent_null_by_col.items()
            if pct_null < pct_null_threshold
        ]

    return _apply_feature_selection(keep, feature_matrix, features)


def remove_single_value_features(
    feature_matrix,
    features=None,
    count_nan_as_value=False,
):
    """Removes columns in feature matrix where all the values are the same.

    Args:
        feature_matrix (:class:`pd.DataFrame`): DataFrame whose columns are feature names and rows are instances.
        features (list[:class:`featuretools.FeatureBase`] or list[str], optional): List of features to select.
        count_nan_as_value (bool): If True, missing values will be counted as their own unique value.
                    If set to False, a feature that has one unique value and all other
                    data missing will be removed from the feature matrix. Defaults to False.

     Returns:
        pd.DataFrame, list[:class:`.FeatureBase`]:
            The feature matrix and the list of generated feature definitions.
            Matches dfs output.
            If no feature list is provided as input, the feature list will not be returned.
    """
    unique_counts_by_col = feature_matrix.nunique(
        dropna=not count_nan_as_value,
    ).to_dict()

    keep = [
        f_name
        for f_name, unique_count in unique_counts_by_col.items()
        if unique_count > 1
    ]
    return _apply_feature_selection(keep, feature_matrix, features)


def remove_highly_correlated_features(
    feature_matrix,
    features=None,
    pct_corr_threshold=0.95,
    features_to_check=None,
    features_to_keep=None,
):
    """Removes columns in feature matrix that are highly correlated with another column.

    Note:
        We make the assumption that, for a pair of features, the feature that is further
        right in the feature matrix produced by ``dfs`` is the more complex one.
        The assumption does not hold if the order of columns in the feature
        matrix has changed from what ``dfs`` produces.

    Args:
        feature_matrix (:class:`pd.DataFrame`): DataFrame whose columns are feature
                    names and rows are instances. If Woodwork is not initalized, will
                    perform Woodwork initialization, which may result in slightly different
                    types than those in the original feature matrix created by Featuretools.
        features (list[:class:`featuretools.FeatureBase`] or list[str], optional):
                    List of features to select.
        pct_corr_threshold (float): The correlation threshold to be considered highly
                    correlated. Defaults to 0.95.
        features_to_check (list[str], optional): List of column names to check
                    whether any pairs are highly correlated. Will not check any
                    other columns, meaning the only columns that can be removed
                    are in this list. If null, defaults to checking all columns.
        features_to_keep (list[str], optional): List of colum names to keep even
                    if correlated to another column. If null, all columns will be
                    candidates for removal.

    Returns:
        pd.DataFrame, list[:class:`.FeatureBase`]:
            The feature matrix and the list of generated feature definitions.
            Matches dfs output. If no feature list is provided as input,
            the feature list will not be returned. For consistent results,
            do not change the order of features outputted by dfs.
    """
    if feature_matrix.ww.schema is None:
        feature_matrix.ww.init()

    if pct_corr_threshold < 0 or pct_corr_threshold > 1:
        raise ValueError(
            "pct_corr_threshold must be a float between 0 and 1, inclusive.",
        )

    if features_to_check is None:
        features_to_check = list(feature_matrix.columns)
    else:
        for f_name in features_to_check:
            assert (
                f_name in feature_matrix.columns
            ), "feature named {} is not in feature matrix".format(f_name)

    if features_to_keep is None:
        features_to_keep = []

    to_select = ["numeric", Boolean, BooleanNullable]
    fm = feature_matrix.ww[features_to_check]
    fm_to_check = fm.ww.select(include=to_select)

    dropped = set()
    columns_to_check = fm_to_check.columns
    # When two features are found to be highly correlated,
    # we drop the more complex feature
    # Columns produced later in dfs are more complex
    for i in range(len(columns_to_check) - 1, 0, -1):
        more_complex_name = columns_to_check[i]
        more_complex_col = fm_to_check[more_complex_name]

        # Convert boolean or Int64 column to be float64
        if pd.api.types.is_bool_dtype(more_complex_col) or isinstance(
            more_complex_col.dtype,
            pd.Int64Dtype,
        ):
            more_complex_col = more_complex_col.astype("float64")

        for j in range(i - 1, -1, -1):
            less_complex_name = columns_to_check[j]
            less_complex_col = fm_to_check[less_complex_name]

            # Convert boolean or Int64 column to be float64
            if pd.api.types.is_bool_dtype(less_complex_col) or isinstance(
                less_complex_col.dtype,
                pd.Int64Dtype,
            ):
                less_complex_col = less_complex_col.astype("float64")

            if abs(more_complex_col.corr(less_complex_col)) >= pct_corr_threshold:
                dropped.add(more_complex_name)
                break

    keep = [
        f_name
        for f_name in feature_matrix.columns
        if (f_name in features_to_keep or f_name not in dropped)
    ]
    return _apply_feature_selection(keep, feature_matrix, features)


def _apply_feature_selection(keep, feature_matrix, features=None):
    new_matrix = feature_matrix[keep]
    new_feature_names = set(new_matrix.columns)

    if features is not None:
        new_features = []
        for f in features:
            if f.number_output_features > 1:
                slices = [
                    f[i]
                    for i in range(f.number_output_features)
                    if f[i].get_name() in new_feature_names
                ]
                if len(slices) == f.number_output_features:
                    new_features.append(f)
                else:
                    new_features.extend(slices)
            else:
                if f.get_name() in new_feature_names:
                    new_features.append(f)

        return new_matrix, new_features
    return new_matrix
