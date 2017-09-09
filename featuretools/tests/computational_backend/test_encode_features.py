from featuretools.synthesis import encode_features
from ..testing_utils import make_ecommerce_entityset
from featuretools.primitives import IdentityFeature
from featuretools import calculate_feature_matrix
import pytest


@pytest.fixture(scope='module')
def entityset():
    return make_ecommerce_entityset()


def test_encodes_features(entityset):
    f1 = IdentityFeature(entityset["log"]["product_id"])
    f2 = IdentityFeature(entityset["log"]["purchased"])
    f3 = IdentityFeature(entityset["log"]["value"])

    features = [f1, f2, f3]
    feature_matrix = calculate_feature_matrix(features, instance_ids=[0, 1, 2, 3, 4, 5])

    feature_matrix_encoded, features_encoded = encode_features(feature_matrix, features)
    assert len(features_encoded) == 6

    feature_matrix_encoded, features_encoded = encode_features(feature_matrix, features, top_n=2)
    assert len(features_encoded) == 5

    feature_matrix_encoded, features_encoded = encode_features(feature_matrix, features,
                                                               include_unknown=False)
    assert len(features_encoded) == 5


def test_inplace_encodes_features(entityset):
    f1 = IdentityFeature(entityset["log"]["product_id"])

    features = [f1]
    feature_matrix = calculate_feature_matrix(features, instance_ids=[0, 1, 2, 3, 4, 5])

    feature_matrix_shape = feature_matrix.shape
    feature_matrix_encoded, features_encoded = encode_features(feature_matrix, features)
    assert feature_matrix_encoded.shape != feature_matrix_shape
    assert feature_matrix.shape == feature_matrix_shape

    # inplace they should be the same
    feature_matrix_encoded, features_encoded = encode_features(feature_matrix, features, inplace=True)
    assert feature_matrix_encoded.shape == feature_matrix.shape


def test_to_encode_features(entityset):
    f1 = IdentityFeature(entityset["log"]["product_id"])
    f2 = IdentityFeature(entityset["log"]["value"])

    features = [f1, f2]
    feature_matrix = calculate_feature_matrix(features, instance_ids=[0, 1, 2, 3, 4, 5])

    feature_matrix_encoded, features_encoded = encode_features(feature_matrix, features)
    feature_matrix_encoded_shape = feature_matrix_encoded.shape

    # to_encode should keep product_id as a string, and not create 3 additional columns
    to_encode = []
    feature_matrix_encoded, features_encoded = encode_features(feature_matrix, features, to_encode=to_encode)
    assert feature_matrix_encoded_shape != feature_matrix_encoded.shape

    to_encode = ['value']
    feature_matrix_encoded, features_encoded = encode_features(feature_matrix, features, to_encode=to_encode)
    assert feature_matrix_encoded_shape != feature_matrix_encoded.shape
