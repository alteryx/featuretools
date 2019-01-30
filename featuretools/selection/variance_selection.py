from __future__ import division, print_function

import pandas as pd

from featuretools.variable_types import Discrete, Numeric


def plot_feature_variances(feature_matrix,
                           features,
                           low_threshold=None,
                           high_threshold=None,
                           log_plot=True,
                           kind='hist',
                           print_description=True):
    ''' Plot to find a good "knee"/threshold at which to select high variance features
        Only applies to numeric features
        Thresholds indicate coefficient of variation (standard deviation divided by mean)'''
    numeric_features = [f.get_name() for f in features if issubclass(f.variable_type, Numeric)]
    stds = feature_matrix[numeric_features].std(axis=0, skipna=True)
    means = feature_matrix[numeric_features].mean(axis=0, skipna=True)
    cvs = (stds / means).abs()
    xlim = [0, cvs.max()]
    if low_threshold is not None:
        cvs = cvs[cvs > low_threshold]
        xlim[0] = low_threshold
    if high_threshold is not None:
        cvs = cvs[cvs < high_threshold]
        xlim[1] = high_threshold
    if print_description:
        print("Stats about the range of the coefficient of variation across all features")
        print(cvs.describe())
    cvs.plot(kind=kind, xlim=tuple(xlim), logx=log_plot)
    return cvs


def get_categorical_nunique_ratio(df, drop_nonumeric=True):
    if drop_nonumeric:
        numeric_columns = df.head()._get_numeric_data().columns
        nonnumeric_columns = [f for f in df if f not in numeric_columns]
        df = df[nonnumeric_columns]
    else:
        nonnumeric_columns = df.columns

    nunique = df.nunique(axis=0, dropna=True)
    total = df[nonnumeric_columns].count(axis=0)
    return nunique / total


def plot_categorical_nunique_ratio(feature_matrix,
                                   low_threshold=None,
                                   high_threshold=None,
                                   log_plot=False,
                                   print_description=True):
    ''' Plot to find a good "knee"/threshold at which to select categorical features
        with a high ratio of nunique to total elements'''
    ratio = get_categorical_nunique_ratio(feature_matrix)

    if low_threshold is not None:
        ratio = ratio[ratio > low_threshold]
    if high_threshold is not None:
        ratio = ratio[ratio < high_threshold]
    if print_description:
        print(ratio.describe())
    return ratio.plot(kind='kde', logx=log_plot)


def select_high_variance_features(feature_matrix, features=None,
                                  cv_threshold=0,
                                  categorical_nunique_ratio=None, keep=None):
    '''
    Select features above a threshold coefficient of variation
    (standard deviation divided by mean).
    By default excludes any non-numeric features. If
    categorical_nunique_ratio is specified, will
    select categorical features whose ratio of unique
    elements to total number of nonnull elements
    is greater than categorical_nunique_ratio

    Args:
        feature_matrix (:class:`pd.DataFrame`): DataFrame whose columns are feature names and rows are instances
        features (list[:class:`featuretools.FeatureBase`] or list[str], optional): List of features to select
        cv_threshold (float): Select features above this coefficient of variation
        categorical_nunique_ratio (float): Select categorical features whose ratio of unique
                elements to total number of nonnull elements is greater than this parameter
        keep (list[str]): list of feature names to force select
    '''
    keep = keep or []
    if features:
        numeric_features = [f.get_name() for f in features if issubclass(f.variable_type, Numeric)]
        stds = feature_matrix[numeric_features].std(axis=0, skipna=True)
        means = feature_matrix[numeric_features].mean(axis=0, skipna=True)
    else:
        stds = feature_matrix.std(axis=0, skipna=True, numeric_only=True)
        means = feature_matrix.mean(axis=0, skipna=True, numeric_only=True)
    cvs = stds / means
    high_variances = cvs[cvs.abs() > cv_threshold]
    if features is None:
        high_variance_feature_names = [f for f in feature_matrix.columns if f in high_variances or f in keep]
    else:
        high_variance_features = [f for f in features if f.get_name() in high_variances.index or f.get_name() in keep]
        high_variance_feature_names = [f.get_name() for f in high_variance_features]

    high_variance_feature_matrix = feature_matrix[high_variance_feature_names]
    if categorical_nunique_ratio is not None:
        if features is not None:
            discrete_features = [f.get_name() for f in features if issubclass(f.variable_type, Discrete)]
            ratio = get_categorical_nunique_ratio(feature_matrix[discrete_features], drop_nonumeric=False)
        else:
            ratio = get_categorical_nunique_ratio(feature_matrix)

        high_ratio = ratio[ratio > categorical_nunique_ratio]
        if features is None:
            high_cat_feature_names = [f for f in feature_matrix if f in high_ratio.index]
        else:
            high_cat_features = [f for f in features if f.get_name() in high_ratio.index]
            high_cat_feature_names = [f.get_name() for f in high_cat_features]
            high_variance_features += high_cat_features
        high_cat_fm = feature_matrix[high_cat_feature_names]
        high_variance_feature_matrix = pd.concat([high_variance_feature_matrix, high_cat_fm], axis=1)
    if features is None:
        return high_variance_feature_matrix
    else:
        return high_variance_feature_matrix, high_variance_features


def select_percent_null(feature_matrix, features, max_null_percent=1.0, keep=None):
    '''Select features where the percentage of null values is below max_null_percent

    Args:
        feature_matrix (:class:`pd.DataFrame`): DataFrame whose columns are feature names and rows are instances
        features (list[:class:`featuretools.FeatureBase`] or list[str], optional): List of features to select
        max_null_percent (float): Select features below this
        keep (list[str]): list of feature names to force select
    '''
    keep = keep or []

    null_counts = feature_matrix.isnull().sum()
    null_percents = null_counts / feature_matrix.shape[0]

    low_nulls = null_percents[null_percents < max_null_percent]

    low_nulls_features = [f for f in features if f.get_name() in low_nulls.index or f.get_name() in keep]
    low_nulls_feature_names = [f.get_name() for f in low_nulls_features]

    return feature_matrix[low_nulls_feature_names], low_nulls_features
