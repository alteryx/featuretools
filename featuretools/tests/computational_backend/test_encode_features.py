import pandas as pd
import pytest

from ..testing_utils import make_ecommerce_entityset

from featuretools import EntitySet, calculate_feature_matrix, dfs
from featuretools.feature_base import Feature, IdentityFeature
from featuretools.primitives import NMostCommon
from featuretools.synthesis import encode_features


@pytest.fixture(scope='module')
def entityset():
    return make_ecommerce_entityset()


def test_encodes_features(entityset):
    f1 = IdentityFeature(entityset["log"]["product_id"])
    f2 = IdentityFeature(entityset["log"]["purchased"])
    f3 = IdentityFeature(entityset["log"]["value"])

    features = [f1, f2, f3]
    feature_matrix = calculate_feature_matrix(features, entityset, instance_ids=[0, 1, 2, 3, 4, 5])

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
    feature_matrix = calculate_feature_matrix(features, entityset, instance_ids=[0, 1, 2, 3, 4, 5])

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
    feature_matrix = calculate_feature_matrix(features, entityset, instance_ids=[0, 1, 2, 3, 4, 5])

    feature_matrix_encoded, features_encoded = encode_features(feature_matrix, features)
    feature_matrix_encoded_shape = feature_matrix_encoded.shape

    # to_encode should keep product_id as a string, and not create 3 additional columns
    to_encode = []
    feature_matrix_encoded, features_encoded = encode_features(feature_matrix, features, to_encode=to_encode)
    assert feature_matrix_encoded_shape != feature_matrix_encoded.shape

    to_encode = ['value']
    feature_matrix_encoded, features_encoded = encode_features(feature_matrix, features, to_encode=to_encode)
    assert feature_matrix_encoded_shape != feature_matrix_encoded.shape


def test_encode_features_handles_pass_columns(entityset):
    f1 = IdentityFeature(entityset["log"]["product_id"])
    f2 = IdentityFeature(entityset["log"]["value"])

    features = [f1, f2]
    cutoff_time = pd.DataFrame({'instance_id': range(6),
                                'time': entityset['log'].df['datetime'][0:6],
                                'label': [i % 2 for i in range(6)]},
                               columns=["instance_id", "time", "label"])
    feature_matrix = calculate_feature_matrix(features, entityset, cutoff_time)

    assert 'label' in feature_matrix.columns

    feature_matrix_encoded, features_encoded = encode_features(feature_matrix, features)
    feature_matrix_encoded_shape = feature_matrix_encoded.shape

    # to_encode should keep product_id as a string, and not create 3 additional columns
    to_encode = []
    feature_matrix_encoded, features_encoded = encode_features(feature_matrix, features, to_encode=to_encode)
    assert feature_matrix_encoded_shape != feature_matrix_encoded.shape

    to_encode = ['value']
    feature_matrix_encoded, features_encoded = encode_features(feature_matrix, features, to_encode=to_encode)
    assert feature_matrix_encoded_shape != feature_matrix_encoded.shape

    assert 'label' in feature_matrix_encoded.columns


def test_encode_features_catches_features_mismatch(entityset):
    f1 = IdentityFeature(entityset["log"]["product_id"])
    f2 = IdentityFeature(entityset["log"]["value"])
    f3 = IdentityFeature(entityset["log"]["session_id"])

    features = [f1, f2]
    cutoff_time = pd.DataFrame({'instance_id': range(6),
                                'time': entityset['log'].df['datetime'][0:6],
                                'label': [i % 2 for i in range(6)]},
                               columns=["instance_id", "time", "label"])
    feature_matrix = calculate_feature_matrix(features, entityset, cutoff_time)

    assert 'label' in feature_matrix.columns

    error_text = 'Feature session_id not found in feature matrix'
    with pytest.raises(AssertionError, match=error_text):
        encode_features(feature_matrix, [f1, f3])


def test_encode_unknown_features():
    # Dataframe with categorical column with "unknown" string
    df = pd.DataFrame({'category': ['unknown', 'b', 'c', 'd', 'e']})

    es = EntitySet('test')
    es.entity_from_dataframe(entity_id='a', dataframe=df, index='index', make_index=True)
    features, feature_defs = dfs(entityset=es, target_entity='a')

    # Specify unknown token for replacement
    features_enc, feature_defs_enc = encode_features(features, feature_defs,
                                                     include_unknown=True)
    assert list(features_enc.columns) == ['category = unknown', 'category = e', 'category = d',
                                          'category = c', 'category = b', 'category is unknown']


def test_encode_features_topn(entityset):
    topn = Feature(entityset['log']['product_id'],
                   parent_entity=entityset['customers'],
                   primitive=NMostCommon(n=3))
    features, feature_defs = dfs(entityset=entityset,
                                 instance_ids=[0, 1, 2],
                                 target_entity="customers",
                                 agg_primitives=[NMostCommon(n=3)])
    features_enc, feature_defs_enc = encode_features(features,
                                                     feature_defs,
                                                     include_unknown=True)
    assert topn.hash() in [feat.hash() for feat in feature_defs_enc]
    for name in topn.get_feature_names():
        assert name in features_enc.columns
