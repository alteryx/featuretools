def remove_low_information_features(feature_matrix, features=None):
    """Select features that have at least 2 unique values and that are not all null

        Args:
            feature_matrix (:class:`pd.DataFrame`): DataFrame whose columns are feature names and rows are instances
            features (list[:class:`featuretools.FeatureBase`] or list[str], optional): List of features to select

        Returns:
            (feature_matrix, features)

    """
    keep = [c for c in feature_matrix
            if (feature_matrix[c].nunique(dropna=False) > 1 and
                feature_matrix[c].dropna().shape[0] > 0)]
    feature_matrix = feature_matrix[keep]
    if features is not None:
        features = [f for f in features
                    if f.get_name() in feature_matrix.columns]
        return feature_matrix, features
    return feature_matrix


'''
The three functions below use logic from EvalML DataChecks
'''


def remove_highly_null_features(feature_matrix, features=None, pct_null_threshold=0.95):
    """
        Determine features from a feature matrix that have higher than a set threshold
        of null values.

        Args:
            feature_matrix (:class:`pd.DataFrame`): DataFrame whose columns are feature names and rows are instances
            features (list[:class:`featuretools.FeatureBase`] or list[str], optional): List of features to select
            pct_null_threshold (float): If the percentage of NaN values in an input feature exceeds this amount,
                    that feature will be considered highly-null. Defaults to 0.95.

        Returns:
            feature matrix (pd.DataFrame): The feature matrix generated. If `features_only` is ``True``,
                the feature matrix will not be generated.
            features (list[:class:`.FeatureBase`]): The list of generated feature defintions.
    """
    if pct_null_threshold < 0 or pct_null_threshold > 1:
        raise ValueError("pct_null_threshold must be a float between 0 and 1, inclusive.")

    percent_null_by_col = (feature_matrix.isnull().mean()).to_dict()

    if pct_null_threshold == 0.0:
        keep = [f_name for f_name, pct_null in percent_null_by_col.items() if pct_null <= pct_null_threshold]
    else:
        keep = [f_name for f_name, pct_null in percent_null_by_col.items() if pct_null < pct_null_threshold]

    return _apply_feature_selection(keep, feature_matrix, features)


def remove_single_value_features(feature_matrix, features=None, count_nan_as_value=False):
    """
        Determines columns in feature matrix where all the values are the same.

        Args:
            feature_matrix (:class:`pd.DataFrame`): DataFrame whose columns are feature names and rows are instances
            features (list[:class:`featuretools.FeatureBase`] or list[str], optional): List of features to select
            count_nan_as_value (bool): If True, missing values will be counted as their own unique value.
                        If set to True, a feature that has one unique value and all other data is missing will be
                        counted as only having a single unique value. Defaults to False.

        Returns:
            feature matrix (pd.DataFrame): The feature matrix generated. If `features_only` is ``True``,
                the feature matrix will not be generated.
            features (list[:class:`.FeatureBase`]): The list of generated feature defintions.
    """
    unique_counts_by_col = feature_matrix.nunique(dropna=not count_nan_as_value).to_dict()

    keep = [f_name for f_name, unique_count in unique_counts_by_col.items() if unique_count > 1]
    return _apply_feature_selection(keep, feature_matrix, features)


def remove_highly_correlated_features(feature_matrix, features=None, pct_corr_threshold=0.95, features_to_check=None, features_to_keep=None):
    """
        Determines whether any pairs of features are highly correlated with one another.

        Args:
            feature_matrix (:class:`pd.DataFrame`): DataFrame whose columns are feature names and rows are instances
            features (list[:class:`featuretools.FeatureBase`] or list[str], optional): List of features to select
            pct_corr_threshold (float): The correlation threshold to be considered highly correlated. Defaults to 0.95.
            features_to_check (list[str], optional): List of column names to check whether any pairs are highly correlated.
                        Will not check any other columns.
                        If null, defaults to checking all columns.
            features_to_keep (list[str], optional): List of colum names to keep even if correlated to another column.
                        If null, all columns will be candidates for removal

        Returns:
            feature matrix (pd.DataFrame): The feature matrix generated. If `features_only` is ``True``,
                the feature matrix will not be generated.
            features (list[:class:`.FeatureBase`]): The list of generated feature defintions.
    """
    if pct_corr_threshold < 0 or pct_corr_threshold > 1:
        raise ValueError("pct_corr_threshold must be a float between 0 and 1, inclusive.")

    # --> consider working more with sets if this is gonna be slow
    if features_to_check is None:
        features_to_check = feature_matrix.columns
    else:
        for f_name in features_to_check:
            assert f_name in feature_matrix.columns, "feature named {} is not in feature matrix".format(f_name)

    if features_to_keep is None:
        features_to_keep = []

    numeric_dtypes = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    boolean = ['bool']
    numeric_and_boolean_dtypes = numeric_dtypes + boolean

    fm_to_check = (feature_matrix[features_to_check]).select_dtypes(include=numeric_and_boolean_dtypes)

    dropped = set()
    # Get all pairs of columns and calculate their correlation
    for f_name1, col1 in fm_to_check.iteritems():
        for f_name2, col2 in fm_to_check.iteritems():
            if f_name1 == f_name2 or f_name1 in dropped or f_name2 in dropped:
                continue

            if abs(col1.corr(col2)) >= pct_corr_threshold:
                dropped.add(f_name1)
                dropped.add(f_name2)

    keep = [f_name for f_name in feature_matrix.columns if (f_name in features_to_keep or
                                                            f_name not in dropped)]
    return _apply_feature_selection(keep, feature_matrix, features)


def _apply_feature_selection(keep, feature_matrix, features=None):
    new_matrix = feature_matrix[keep]
    new_feature_names = set(new_matrix.columns)

    if features is not None:
        features = [f for f in features if f.get_name() in new_feature_names]
        return new_matrix, features
    return new_matrix
