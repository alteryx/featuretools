from featuretools.selection import select_high_variance_features
from featuretools.tests.testing_utils import make_ecommerce_entityset
from featuretools import Feature
import pandas as pd
import pytest


@pytest.fixture(scope='module')
def feature_matrix():
    feature_matrix = pd.DataFrame({'numeric_low': [0, 0, 0],
                                   'numeric_high': [0, 100, 200],
                                   'numeric_high_low_cv': [9500, 10000, 10500],
                                   'categorical_low': ['test', 'test', 'test2'],
                                   'categorical_high': ['test1', 'test2', 'test3']})
    return feature_matrix


@pytest.fixture(scope='module')
def es(feature_matrix):
    es = make_ecommerce_entityset()
    es.entity_from_dataframe('test', feature_matrix, index='test', make_index=True)
    return es


def test_select_high_variance_feature_names(feature_matrix):
    feature_matrix = select_high_variance_features(feature_matrix, cv_threshold=0.5, categorical_nunique_ratio=.7)
    assert feature_matrix.shape == (3, 2)
    assert 'numeric_high' in feature_matrix.columns
    assert 'categorical_high' in feature_matrix.columns


def test_select_high_variance_features(es, feature_matrix):
    features = [Feature(v) for v in es['test'].variables]
    feature_matrix, features = select_high_variance_features(feature_matrix, features, cv_threshold=0.5, categorical_nunique_ratio=.7)
    assert feature_matrix.shape == (3, 2)
    assert len(features) == 2
    for f in features:
        assert f.get_name() in feature_matrix.columns

    assert 'numeric_high' in feature_matrix.columns
    assert 'categorical_high' in feature_matrix.columns
