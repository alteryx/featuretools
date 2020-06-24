import os

import boto3
import pytest
from pympler.asizeof import asizeof
from smart_open import open

import featuretools as ft
from featuretools.feature_base.features_deserializer import (
    FeaturesDeserializer
)
from featuretools.feature_base.features_serializer import FeaturesSerializer
from featuretools.primitives import (
    Count,
    CumSum,
    Day,
    Haversine,
    Max,
    Mean,
    Min,
    Mode,
    Month,
    NMostCommon,
    NumCharacters,
    NumUnique,
    NumWords,
    PercentTrue,
    Skew,
    Std,
    Sum,
    Weekday,
    Year,
    make_agg_primitive
)
from featuretools.tests.testing_utils import check_names
from featuretools.variable_types import Numeric

BUCKET_NAME = "test-bucket"
WRITE_KEY_NAME = "test-key"
TEST_S3_URL = "s3://{}/{}".format(BUCKET_NAME, WRITE_KEY_NAME)
TEST_FILE = "test_feature_serialization_feature_schema_5.0.0_entityset_schema_4.0.0.json"
S3_URL = "s3://featuretools-static/" + TEST_FILE
URL = "https://featuretools-static.s3.amazonaws.com/" + TEST_FILE
TEST_CONFIG = "CheckConfigPassesOn"
TEST_KEY = "test_access_key_features"


def assert_features(original, deserialized):
    for feat_1, feat_2 in zip(original, deserialized):
        assert feat_1.unique_name() == feat_2.unique_name()
        assert feat_1.entityset == feat_2.entityset


def pickle_features_test_helper(es_size, features_original, dir_path):
    filepath = os.path.join(dir_path, 'test_feature')

    ft.save_features(features_original, filepath)
    features_deserializedA = ft.load_features(filepath)
    assert os.path.getsize(filepath) < es_size
    os.remove(filepath)

    with open(filepath, "w") as f:
        ft.save_features(features_original, f)
    features_deserializedB = ft.load_features(open(filepath))
    assert os.path.getsize(filepath) < es_size
    os.remove(filepath)

    features = ft.save_features(features_original)
    features_deserializedC = ft.load_features(features)
    assert asizeof(features) < es_size

    features_deserialized_options = [features_deserializedA, features_deserializedB, features_deserializedC]
    for features_deserialized in features_deserialized_options:
        assert_features(features_original, features_deserialized)


def test_pickle_features(es, tmpdir):
    features_original = ft.dfs(target_entity='sessions', entityset=es, features_only=True)
    pickle_features_test_helper(asizeof(es), features_original, str(tmpdir))


def test_pickle_features_with_custom_primitive(pd_es, tmpdir):
    NewMax = make_agg_primitive(
        lambda x: max(x),
        name="NewMax",
        input_types=[Numeric],
        return_type=Numeric,
        description="Calculate means ignoring nan values")

    features_original = ft.dfs(target_entity='sessions', entityset=pd_es,
                               agg_primitives=["Last", "Mean", NewMax], features_only=True)

    assert any([isinstance(feat.primitive, NewMax) for feat in features_original])
    pickle_features_test_helper(asizeof(pd_es), features_original, str(tmpdir))


def test_serialized_renamed_features(es):
    def serialize_name_unchanged(original):
        new_name = 'MyFeature'
        original_names = original.get_feature_names()
        renamed = original.rename(new_name)
        new_names = [new_name] if len(original_names) == 1 else [new_name + '[{}]'.format(i) for i in range(len(original_names))]
        check_names(renamed, new_name, new_names)

        serializer = FeaturesSerializer([renamed])
        serialized = serializer.to_dict()

        deserializer = FeaturesDeserializer(serialized)
        deserialized = deserializer.to_list()[0]
        check_names(deserialized, new_name, new_names)

    identity_original = ft.IdentityFeature(es['log']['value'])
    assert identity_original.get_name() == 'value'

    value = ft.IdentityFeature(es['log']['value'])

    primitive = ft.primitives.Max()
    agg_original = ft.AggregationFeature(value, es['customers'], primitive)
    assert agg_original.get_name() == 'MAX(log.value)'

    direct_original = ft.DirectFeature(es['customers']['age'], es['sessions'])
    assert direct_original.get_name() == 'customers.age'

    primitive = ft.primitives.MultiplyNumericScalar(value=2)
    transform_original = ft.TransformFeature(value, primitive)
    assert transform_original.get_name() == 'value * 2'

    zipcode = ft.IdentityFeature(es['log']['zipcode'])
    primitive = CumSum()
    groupby_original = ft.feature_base.GroupByTransformFeature(value, primitive, zipcode)
    assert groupby_original.get_name() == 'CUM_SUM(value) by zipcode'

    multioutput_original = ft.Feature(es['log']['product_id'], parent_entity=es['customers'], primitive=NMostCommon(n=2))
    assert multioutput_original.get_name() == 'N_MOST_COMMON(log.product_id, n=2)'

    featureslice_original = ft.feature_base.FeatureOutputSlice(multioutput_original, 0)
    assert featureslice_original.get_name() == 'N_MOST_COMMON(log.product_id, n=2)[0]'

    feature_type_list = [identity_original, agg_original, direct_original, transform_original, groupby_original, multioutput_original, featureslice_original]

    for feature_type in feature_type_list:
        serialize_name_unchanged(feature_type)


@pytest.fixture
def s3_client():
    _environ = os.environ.copy()
    from moto import mock_s3
    with mock_s3():
        s3 = boto3.resource('s3')
        yield s3
    os.environ.clear()
    os.environ.update(_environ)


@pytest.fixture
def s3_bucket(s3_client):
    s3_client.create_bucket(Bucket=BUCKET_NAME, ACL='public-read-write')
    s3_bucket = s3_client.Bucket(BUCKET_NAME)
    yield s3_bucket


def test_serialize_features_mock_s3(es, s3_client, s3_bucket):
    features_original = ft.dfs(target_entity='sessions', entityset=es, features_only=True)

    ft.save_features(features_original, TEST_S3_URL)

    obj = list(s3_bucket.objects.all())[0].key
    s3_client.ObjectAcl(BUCKET_NAME, obj).put(ACL='public-read-write')

    features_deserialized = ft.load_features(TEST_S3_URL)
    assert_features(features_original, features_deserialized)


def test_serialize_features_mock_anon_s3(es, s3_client, s3_bucket):
    features_original = ft.dfs(target_entity='sessions', entityset=es, features_only=True)

    ft.save_features(features_original, TEST_S3_URL, profile_name=False)

    obj = list(s3_bucket.objects.all())[0].key
    s3_client.ObjectAcl(BUCKET_NAME, obj).put(ACL='public-read-write')

    features_deserialized = ft.load_features(TEST_S3_URL, profile_name=False)
    assert_features(features_original, features_deserialized)


@pytest.fixture
def setup_test_profile(monkeypatch, tmpdir):
    cache = str(tmpdir.join('.cache').mkdir())
    test_path = os.path.join(cache, 'test_credentials')
    test_path_config = os.path.join(cache, 'test_config')
    monkeypatch.setenv("AWS_SHARED_CREDENTIALS_FILE", test_path)
    monkeypatch.setenv("AWS_CONFIG_FILE", test_path_config)
    monkeypatch.delenv("AWS_ACCESS_KEY_ID", raising=False)
    monkeypatch.delenv("AWS_SECRET_ACCESS_KEY", raising=False)
    monkeypatch.setenv("AWS_PROFILE", "test")

    try:
        os.remove(test_path)
        os.remove(test_path_config)
    except EnvironmentError:
        pass

    f = open(test_path, "w+")
    f.write("[test]\n")
    f.write("aws_access_key_id=AKIAIOSFODNN7EXAMPLE\n")
    f.write("aws_secret_access_key=wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY\n")
    f.close()
    f = open(test_path_config, "w+")
    f.write("[profile test]\n")
    f.write("region=us-east-2\n")
    f.write("output=text\n")
    f.close()

    yield
    os.remove(test_path)
    os.remove(test_path_config)


def test_s3_test_profile(es, s3_client, s3_bucket, setup_test_profile):
    features_original = ft.dfs(target_entity='sessions', entityset=es, features_only=True)

    ft.save_features(features_original, TEST_S3_URL, profile_name='test')

    obj = list(s3_bucket.objects.all())[0].key
    s3_client.ObjectAcl(BUCKET_NAME, obj).put(ACL='public-read-write')

    features_deserialized = ft.load_features(TEST_S3_URL, profile_name='test')
    assert_features(features_original, features_deserialized)


@pytest.mark.parametrize("url,profile_name", [(S3_URL, None), (S3_URL, False),
                                              (URL, None)])
def test_deserialize_features_s3(pd_es, url, profile_name):
    agg_primitives = [Sum, Std, Max, Skew, Min, Mean, Count, PercentTrue,
                      NumUnique, Mode]

    trans_primitives = [Day, Year, Month, Weekday, Haversine, NumWords,
                        NumCharacters]

    features_original = sorted(ft.dfs(target_entity='sessions',
                                      entityset=pd_es,
                                      features_only=True,
                                      agg_primitives=agg_primitives,
                                      trans_primitives=trans_primitives),
                               key=lambda x: x.unique_name())
    features_deserialized = sorted(ft.load_features(url,
                                                    profile_name=profile_name),
                                   key=lambda x: x.unique_name())
    assert_features(features_original, features_deserialized)


def test_serialize_url(es):
    features_original = ft.dfs(target_entity='sessions',
                               entityset=es,
                               features_only=True)
    error_text = "Writing to URLs is not supported"
    with pytest.raises(ValueError, match=error_text):
        ft.save_features(features_original, URL)
