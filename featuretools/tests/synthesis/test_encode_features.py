import pandas as pd
import pytest

from featuretools import EntitySet, calculate_feature_matrix, dfs
from featuretools.feature_base import Feature, IdentityFeature
from featuretools.primitives import NMostCommon
from featuretools.synthesis import encode_features


def test_encodes_features(pd_es):
    f1 = IdentityFeature(pd_es["log"]["product_id"])
    f2 = IdentityFeature(pd_es["log"]["purchased"])
    f3 = IdentityFeature(pd_es["log"]["value"])

    features = [f1, f2, f3]
    feature_matrix = calculate_feature_matrix(features, pd_es, instance_ids=[0, 1, 2, 3, 4, 5])

    feature_matrix_encoded, features_encoded = encode_features(feature_matrix, features)
    assert len(features_encoded) == 6

    feature_matrix_encoded, features_encoded = encode_features(feature_matrix, features, top_n=2)
    assert len(features_encoded) == 5

    feature_matrix_encoded, features_encoded = encode_features(feature_matrix, features,
                                                               include_unknown=False)
    assert len(features_encoded) == 5


def test_dask_errors_encode_features(dask_es):
    f1 = IdentityFeature(dask_es["log"]["product_id"])
    f2 = IdentityFeature(dask_es["log"]["purchased"])
    f3 = IdentityFeature(dask_es["log"]["value"])

    features = [f1, f2, f3]
    feature_matrix = calculate_feature_matrix(features,
                                              dask_es,
                                              instance_ids=[0, 1, 2, 3, 4, 5])
    error_text = "feature_matrix must be a Pandas DataFrame"

    with pytest.raises(TypeError, match=error_text):
        encode_features(feature_matrix, features)


def test_inplace_encodes_features(pd_es):
    f1 = IdentityFeature(pd_es["log"]["product_id"])

    features = [f1]
    feature_matrix = calculate_feature_matrix(features, pd_es, instance_ids=[0, 1, 2, 3, 4, 5])

    feature_matrix_shape = feature_matrix.shape
    feature_matrix_encoded, features_encoded = encode_features(feature_matrix, features)
    assert feature_matrix_encoded.shape != feature_matrix_shape
    assert feature_matrix.shape == feature_matrix_shape

    # inplace they should be the same
    feature_matrix_encoded, features_encoded = encode_features(feature_matrix, features, inplace=True)
    assert feature_matrix_encoded.shape == feature_matrix.shape


def test_to_encode_features(pd_es):
    f1 = IdentityFeature(pd_es["log"]["product_id"])
    f2 = IdentityFeature(pd_es["log"]["value"])
    f3 = IdentityFeature(pd_es["log"]["datetime"])

    features = [f1, f2, f3]
    feature_matrix = calculate_feature_matrix(features, pd_es, instance_ids=[0, 1, 2, 3, 4, 5])

    feature_matrix_encoded, features_encoded = encode_features(feature_matrix, features)
    feature_matrix_encoded_shape = feature_matrix_encoded.shape

    # to_encode should keep product_id as a string and datetime as a date,
    # and not have the same shape as previous encoded matrix due to fewer encoded features
    to_encode = []
    feature_matrix_encoded, features_encoded = encode_features(feature_matrix, features, to_encode=to_encode)
    assert feature_matrix_encoded_shape != feature_matrix_encoded.shape
    assert feature_matrix_encoded['datetime'].dtype == "datetime64[ns]"
    assert feature_matrix_encoded['product_id'].dtype == "object"

    to_encode = ['value']
    feature_matrix_encoded, features_encoded = encode_features(feature_matrix, features, to_encode=to_encode)
    assert feature_matrix_encoded_shape != feature_matrix_encoded.shape
    assert feature_matrix_encoded['datetime'].dtype == "datetime64[ns]"
    assert feature_matrix_encoded['product_id'].dtype == "object"


def test_encode_features_handles_pass_columns(pd_es):
    f1 = IdentityFeature(pd_es["log"]["product_id"])
    f2 = IdentityFeature(pd_es["log"]["value"])

    features = [f1, f2]
    cutoff_time = pd.DataFrame({'instance_id': range(6),
                                'time': pd_es['log'].df['datetime'][0:6],
                                'label': [i % 2 for i in range(6)]},
                               columns=["instance_id", "time", "label"])
    feature_matrix = calculate_feature_matrix(features, pd_es, cutoff_time)

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


def test_encode_features_catches_features_mismatch(pd_es):
    f1 = IdentityFeature(pd_es["log"]["product_id"])
    f2 = IdentityFeature(pd_es["log"]["value"])
    f3 = IdentityFeature(pd_es["log"]["session_id"])

    features = [f1, f2]
    cutoff_time = pd.DataFrame({'instance_id': range(6),
                                'time': pd_es['log'].df['datetime'][0:6],
                                'label': [i % 2 for i in range(6)]},
                               columns=["instance_id", "time", "label"])
    feature_matrix = calculate_feature_matrix(features, pd_es, cutoff_time)

    assert 'label' in feature_matrix.columns

    error_text = 'Feature session_id not found in feature matrix'
    with pytest.raises(AssertionError, match=error_text):
        encode_features(feature_matrix, [f1, f3])


def test_encode_unknown_features():
    # Dataframe with categorical column with "unknown" string
    df = pd.DataFrame({'category': ['unknown', 'b', 'c', 'd', 'e']})

    pd_es = EntitySet('test')
    pd_es.entity_from_dataframe(entity_id='a', dataframe=df, index='index', make_index=True)
    features, feature_defs = dfs(entityset=pd_es, target_entity='a')

    # Specify unknown token for replacement
    features_enc, feature_defs_enc = encode_features(features, feature_defs,
                                                     include_unknown=True)
    assert list(features_enc.columns) == ['category = unknown', 'category = e', 'category = d',
                                          'category = c', 'category = b', 'category is unknown']


def test_encode_features_topn(pd_es):
    topn = Feature(pd_es['log']['product_id'],
                   parent_entity=pd_es['customers'],
                   primitive=NMostCommon(n=3))
    features, feature_defs = dfs(entityset=pd_es,
                                 instance_ids=[0, 1, 2],
                                 target_entity="customers",
                                 agg_primitives=[NMostCommon(n=3)])
    features_enc, feature_defs_enc = encode_features(features,
                                                     feature_defs,
                                                     include_unknown=True)
    assert topn.unique_name() in [feat.unique_name() for feat in feature_defs_enc]
    for name in topn.get_feature_names():
        assert name in features_enc.columns
        assert features_enc.columns.tolist().count(name) == 1


def test_encode_features_drop_first():
    df = pd.DataFrame({'category': ['ao', 'b', 'c', 'd', 'e']})
    pd_es = EntitySet('test')
    pd_es.entity_from_dataframe(entity_id='a', dataframe=df, index='index', make_index=True)
    features, feature_defs = dfs(entityset=pd_es, target_entity='a')
    features_enc, feature_defs_enc = encode_features(features, feature_defs,
                                                     drop_first=True, include_unknown=False)
    assert len(features_enc.columns) == 4

    features_enc, feature_defs = encode_features(features, feature_defs, top_n=3, drop_first=True,
                                                 include_unknown=False)

    assert len(features_enc.columns) == 2


def test_encode_features_handles_dictionary_input(pd_es):
    f1 = IdentityFeature(pd_es["log"]["product_id"])
    f2 = IdentityFeature(pd_es["log"]["purchased"])
    f3 = IdentityFeature(pd_es["log"]["session_id"])

    features = [f1, f2, f3]
    feature_matrix = calculate_feature_matrix(features, pd_es, instance_ids=range(16))
    feature_matrix_encoded, features_encoded = encode_features(feature_matrix, features)
    true_values = ['product_id = coke zero', 'product_id = toothpaste', 'product_id = car',
                   'product_id = brown bag', 'product_id = taco clock', 'product_id = Haribo sugar-free gummy bears',
                   'product_id is unknown', 'purchased', 'session_id = 0', 'session_id = 1', 'session_id = 4',
                   'session_id = 3', 'session_id = 5', 'session_id = 2', 'session_id is unknown']
    assert len(features_encoded) == 15
    for col in true_values:
        assert col in list(feature_matrix_encoded.columns)

    top_n_dict = {}
    feature_matrix_encoded, features_encoded = encode_features(feature_matrix, features, top_n=top_n_dict)
    assert len(features_encoded) == 15
    for col in true_values:
        assert col in list(feature_matrix_encoded.columns)

    top_n_dict = {f1.get_name(): 4, f3.get_name(): 3}
    feature_matrix_encoded, features_encoded = encode_features(feature_matrix, features, top_n=top_n_dict)
    assert len(features_encoded) == 10
    true_values = ['product_id = coke zero', 'product_id = toothpaste', 'product_id = car',
                   'product_id = brown bag', 'product_id is unknown', 'purchased',
                   'session_id = 0', 'session_id = 1', 'session_id = 4', 'session_id is unknown']
    for col in true_values:
        assert col in list(feature_matrix_encoded.columns)

    feature_matrix_encoded, features_encoded = encode_features(feature_matrix, features, top_n=top_n_dict, include_unknown=False)
    true_values = ['product_id = coke zero', 'product_id = toothpaste', 'product_id = car',
                   'product_id = brown bag', 'purchased', 'session_id = 0', 'session_id = 1', 'session_id = 4']
    assert len(features_encoded) == 8
    for col in true_values:
        assert col in list(feature_matrix_encoded.columns)


def test_encode_features_matches_calculate_feature_matrix():
    df = pd.DataFrame({'category': ['b', 'c', 'd', 'e']})

    pd_es = EntitySet('test')
    pd_es.entity_from_dataframe(
        entity_id='a', dataframe=df, index='index', make_index=True)
    features, feature_defs = dfs(entityset=pd_es, target_entity='a')

    features_enc, feature_defs_enc = encode_features(features, feature_defs, to_encode=['category'])

    features_calc = calculate_feature_matrix(feature_defs_enc, entityset=pd_es)

    assert features_enc['category = e'].dtypes == bool
    assert features_enc['category = e'].dtypes == features_calc['category = e'].dtypes
