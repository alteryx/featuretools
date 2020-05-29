import numpy as np
import pandas as pd
import pytest

from featuretools import Feature
from featuretools.selection import remove_low_information_features


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
    feature_matrix = remove_low_information_features(feature_matrix)
    assert feature_matrix.shape == (3, 5)
    assert 'one_value' not in feature_matrix.columns
    assert 'all_null' not in feature_matrix.columns


# remove low information features not supported in Dask
def test_remove_low_information_features(test_es, feature_matrix):
    features = [Feature(v) for v in test_es['test'].variables]
    feature_matrix, features = remove_low_information_features(feature_matrix,
                                                               features)
    assert feature_matrix.shape == (3, 5)
    assert len(features) == 5
    for f in features:
        assert f.get_name() in feature_matrix.columns
    assert 'one_value' not in feature_matrix.columns
    assert 'all_null' not in feature_matrix.columns
