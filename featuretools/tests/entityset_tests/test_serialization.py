import json
import os

import boto3
import pandas as pd
import pytest

from featuretools.demo import load_mock_customer
from featuretools.entityset import EntitySet, deserialize, serialize
from featuretools.variable_types.variable import (
    Categorical,
    Index,
    TimeIndex,
    find_variable_types
)

BUCKET_NAME = "test-bucket"
WRITE_KEY_NAME = "test-key"
TEST_S3_URL = "s3://{}/{}".format(BUCKET_NAME, WRITE_KEY_NAME)
TEST_FILE = "test_serialization_data_entityset_schema_2.0.0.tar"
S3_URL = "s3://featuretools-static/" + TEST_FILE
URL = "https://featuretools-static.s3.amazonaws.com/" + TEST_FILE
TEST_KEY = "test_access_key_es"


def test_all_variable_descriptions():
    variable_types = find_variable_types()
    pd_es = EntitySet()
    dataframe = pd.DataFrame(columns=list(variable_types))
    pd_es.entity_from_dataframe(
        'variable_types',
        dataframe,
        index='index',
        time_index='datetime_time_index',
        variable_types=variable_types,
    )
    entity = pd_es['variable_types']
    for variable in entity.variables:
        description = variable.to_data_description()
        _variable = deserialize.description_to_variable(description, entity=entity)
        assert variable.__eq__(_variable)


def test_custom_variable_descriptions():

    class ItemList(Categorical):
        type_string = "item_list"
        _default_pandas_dtype = list

    pd_es = EntitySet()
    variables = {'item_list': ItemList, 'time_index': TimeIndex, 'index': Index}
    dataframe = pd.DataFrame(columns=list(variables))
    pd_es.entity_from_dataframe(
        'custom_variable', dataframe, index='index',
        time_index='time_index', variable_types=variables)
    entity = pd_es['custom_variable']
    for variable in entity.variables:
        description = variable.to_data_description()
        _variable = deserialize.description_to_variable(description, entity=entity)
        assert variable.__eq__(_variable)


def test_variable_descriptions(pd_es):
    for entity in pd_es.entities:
        for variable in entity.variables:
            description = variable.to_data_description()
            _variable = deserialize.description_to_variable(description, entity=entity)
            assert variable.__eq__(_variable)


def test_entity_descriptions(pd_es):
    _es = EntitySet(pd_es.id)
    for entity in pd_es.metadata.entities:
        description = serialize.entity_to_description(entity)
        deserialize.description_to_entity(description, _es)
        _entity = _es[description['id']]
        _entity.last_time_index = entity.last_time_index
        assert entity.__eq__(_entity, deep=True)


def test_entityset_description(pd_es):
    description = serialize.entityset_to_description(pd_es)
    _es = deserialize.description_to_entityset(description)
    assert pd_es.metadata.__eq__(_es, deep=True)


def test_invalid_formats(pd_es, tmpdir):
    error_text = 'must be one of the following formats: {}'
    error_text = error_text.format(', '.join(serialize.FORMATS))
    with pytest.raises(ValueError, match=error_text):
        serialize.write_entity_data(pd_es.entities[0], path=str(tmpdir), format='')
    with pytest.raises(ValueError, match=error_text):
        entity = {'loading_info': {'location': 'data', 'type': ''}}
        deserialize.read_entity_data(entity, path='.')


def test_empty_dataframe(pd_es):
    for entity in pd_es.entities:
        description = serialize.entity_to_description(entity)
        dataframe = deserialize.empty_dataframe(description)
        assert dataframe.empty


def test_to_csv(pd_es, tmpdir):
    pd_es.to_csv(str(tmpdir), encoding='utf-8', engine='python')
    new_es = deserialize.read_entityset(str(tmpdir))
    assert pd_es.__eq__(new_es, deep=True)
    assert type(pd_es['log'].df['latlong'][0]) == tuple
    assert type(new_es['log'].df['latlong'][0]) == tuple


def test_to_pickle(pd_es, tmpdir):
    pd_es.to_pickle(str(tmpdir))
    new_es = deserialize.read_entityset(str(tmpdir))
    assert pd_es.__eq__(new_es, deep=True)
    assert type(pd_es['log'].df['latlong'][0]) == tuple
    assert type(new_es['log'].df['latlong'][0]) == tuple


def test_to_pickle_interesting_values(pd_es, tmpdir):
    pd_es.add_interesting_values()
    pd_es.to_pickle(str(tmpdir))
    new_es = deserialize.read_entityset(str(tmpdir))
    assert pd_es.__eq__(new_es, deep=True)


def test_to_pickle_manual_interesting_values(pd_es, tmpdir):
    pd_es['log']['product_id'].interesting_values = ["coke_zero"]
    pd_es.to_pickle(str(tmpdir))
    new_es = deserialize.read_entityset(str(tmpdir))
    assert pd_es.__eq__(new_es, deep=True)


def test_to_parquet(pd_es, tmpdir):
    pd_es.to_parquet(str(tmpdir))
    new_es = deserialize.read_entityset(str(tmpdir))
    assert pd_es.__eq__(new_es, deep=True)
    assert type(pd_es['log'].df['latlong'][0]) == tuple
    assert type(new_es['log'].df['latlong'][0]) == tuple


def test_to_parquet_manual_interesting_values(pd_es, tmpdir):
    pd_es['log']['product_id'].interesting_values = ["coke_zero"]
    pd_es.to_pickle(str(tmpdir))
    new_es = deserialize.read_entityset(str(tmpdir))
    assert pd_es.__eq__(new_es, deep=True)


def test_to_parquet_interesting_values(pd_es, tmpdir):
    pd_es.add_interesting_values()
    pd_es.to_parquet(str(tmpdir))
    new_es = deserialize.read_entityset(str(tmpdir))
    assert pd_es.__eq__(new_es, deep=True)


def test_to_parquet_with_lti(tmpdir):
    pd_es = load_mock_customer(return_entityset=True, random_seed=0)
    pd_es.to_parquet(str(tmpdir))
    new_es = deserialize.read_entityset(str(tmpdir))
    assert pd_es.__eq__(new_es, deep=True)


def test_to_pickle_id_none(tmpdir):
    pd_es = EntitySet()
    pd_es.to_pickle(str(tmpdir))
    new_es = deserialize.read_entityset(str(tmpdir))
    assert pd_es.__eq__(new_es, deep=True)

# TODO: Fix Moto tests needing to explicitly set permissions for objects
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


def make_public(s3_client, s3_bucket):
    obj = list(s3_bucket.objects.all())[0].key
    s3_client.ObjectAcl(BUCKET_NAME, obj).put(ACL='public-read-write')


def test_serialize_s3_csv(pd_es, s3_client, s3_bucket):
    pd_es.to_csv(TEST_S3_URL, encoding='utf-8', engine='python')
    make_public(s3_client, s3_bucket)
    new_es = deserialize.read_entityset(TEST_S3_URL)
    assert pd_es.__eq__(new_es, deep=True)


def test_serialize_s3_pickle(pd_es, s3_client, s3_bucket):
    pd_es.to_pickle(TEST_S3_URL)
    make_public(s3_client, s3_bucket)
    new_es = deserialize.read_entityset(TEST_S3_URL)
    assert pd_es.__eq__(new_es, deep=True)


def test_serialize_s3_parquet(pd_es, s3_client, s3_bucket):
    pd_es.to_parquet(TEST_S3_URL)
    make_public(s3_client, s3_bucket)
    new_es = deserialize.read_entityset(TEST_S3_URL)
    assert pd_es.__eq__(new_es, deep=True)


def test_serialize_s3_anon_csv(pd_es, s3_client, s3_bucket):
    pd_es.to_csv(TEST_S3_URL, encoding='utf-8', engine='python', profile_name=False)
    make_public(s3_client, s3_bucket)
    new_es = deserialize.read_entityset(TEST_S3_URL, profile_name=False)
    assert pd_es.__eq__(new_es, deep=True)


def test_serialize_s3_anon_pickle(pd_es, s3_client, s3_bucket):
    pd_es.to_pickle(TEST_S3_URL, profile_name=False)
    make_public(s3_client, s3_bucket)
    new_es = deserialize.read_entityset(TEST_S3_URL, profile_name=False)
    assert pd_es.__eq__(new_es, deep=True)


def test_serialize_s3_anon_parquet(pd_es, s3_client, s3_bucket):
    pd_es.to_parquet(TEST_S3_URL, profile_name=False)
    make_public(s3_client, s3_bucket)
    new_es = deserialize.read_entityset(TEST_S3_URL, profile_name=False)
    assert pd_es.__eq__(new_es, deep=True)


def create_test_credentials(test_path):
    with open(test_path, "w+") as f:
        f.write("[test]\n")
        f.write("aws_access_key_id=AKIAIOSFODNN7EXAMPLE\n")
        f.write("aws_secret_access_key=wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY\n")


def create_test_config(test_path_config):
    with open(test_path_config, "w+") as f:
        f.write("[profile test]\n")
        f.write("region=us-east-2\n")
        f.write("output=text\n")


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
    except OSError:
        pass

    create_test_credentials(test_path)
    create_test_config(test_path_config)
    yield
    os.remove(test_path)
    os.remove(test_path_config)


def test_s3_test_profile(pd_es, s3_client, s3_bucket, setup_test_profile):
    pd_es.to_csv(TEST_S3_URL, encoding='utf-8', engine='python', profile_name='test')
    make_public(s3_client, s3_bucket)
    new_es = deserialize.read_entityset(TEST_S3_URL, profile_name='test')
    assert pd_es.__eq__(new_es, deep=True)


def test_serialize_url_csv(pd_es):
    error_text = "Writing to URLs is not supported"
    with pytest.raises(ValueError, match=error_text):
        pd_es.to_csv(URL, encoding='utf-8', engine='python')


def test_serialize_subdirs_not_removed(pd_es, tmpdir):
    write_path = tmpdir.mkdir("test")
    test_dir = write_path.mkdir("test_dir")
    with open(str(write_path.join('data_description.json')), 'w') as f:
        json.dump('__SAMPLE_TEXT__', f)
    serialize.write_data_description(pd_es, path=str(write_path), index='1', sep='\t', encoding='utf-8', compression=None)
    assert os.path.exists(str(test_dir))
    with open(str(write_path.join('data_description.json')), 'r') as f:
        assert '__SAMPLE_TEXT__' not in json.load(f)


def test_deserialize_url_csv(pd_es):
    new_es = deserialize.read_entityset(URL)
    assert pd_es.__eq__(new_es, deep=True)


def test_default_s3_csv(pd_es):
    new_es = deserialize.read_entityset(S3_URL)
    assert pd_es.__eq__(new_es, deep=True)


def test_anon_s3_csv(pd_es):
    new_es = deserialize.read_entityset(S3_URL, profile_name=False)
    assert pd_es.__eq__(new_es, deep=True)
