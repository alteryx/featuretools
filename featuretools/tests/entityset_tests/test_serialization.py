import errno
import os
import shutil

import boto3
import pandas as pd
import pytest

from featuretools.demo import load_mock_customer
from featuretools.entityset import EntitySet, deserialize, serialize
from featuretools.tests import integration_data
from featuretools.variable_types.variable import (
    Categorical,
    Index,
    TimeIndex,
    find_variable_types
)

CACHE = os.path.join(os.path.dirname(integration_data.__file__), '.cache')
BUCKET_NAME = "test-bucket"
WRITE_KEY_NAME = "test-key"
TEST_S3_URL = "s3://{}/{}".format(BUCKET_NAME, WRITE_KEY_NAME)
S3_URL = "s3://featuretools-static/test_serialization_data_1.0.0.tar"
URL = 'https://featuretools-static.s3.amazonaws.com/test_serialization_data_1.0.0.tar'
TEST_KEY = "test_access_key_es"


def test_all_variable_descriptions():
    variable_types = find_variable_types()
    es = EntitySet()
    dataframe = pd.DataFrame(columns=list(variable_types))
    es.entity_from_dataframe(
        'variable_types',
        dataframe,
        index='index',
        time_index='datetime_time_index',
        variable_types=variable_types,
    )
    entity = es['variable_types']
    for variable in entity.variables:
        description = variable.to_data_description()
        _variable = deserialize.description_to_variable(description, entity=entity)
        assert variable.__eq__(_variable)


def test_custom_variable_descriptions():

    class ItemList(Categorical):
        type_string = "item_list"
        _default_pandas_dtype = list

    es = EntitySet()
    variables = {'item_list': ItemList, 'time_index': TimeIndex, 'index': Index}
    dataframe = pd.DataFrame(columns=list(variables))
    es.entity_from_dataframe(
        'custom_variable', dataframe, index='index',
        time_index='time_index', variable_types=variables)
    entity = es['custom_variable']
    for variable in entity.variables:
        description = variable.to_data_description()
        _variable = deserialize.description_to_variable(description, entity=entity)
        assert variable.__eq__(_variable)


def test_variable_descriptions(es):
    for entity in es.entities:
        for variable in entity.variables:
            description = variable.to_data_description()
            _variable = deserialize.description_to_variable(description, entity=entity)
            assert variable.__eq__(_variable)


def test_entity_descriptions(es):
    _es = EntitySet(es.id)
    for entity in es.metadata.entities:
        description = serialize.entity_to_description(entity)
        deserialize.description_to_entity(description, _es)
        _entity = _es[description['id']]
        _entity.last_time_index = entity.last_time_index
        assert entity.__eq__(_entity, deep=True)


def test_entityset_description(es):
    description = serialize.entityset_to_description(es)
    _es = deserialize.description_to_entityset(description)
    assert es.metadata.__eq__(_es, deep=True)


@pytest.fixture
def path_management():
    path = os.path.join(CACHE, 'es')
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:  # EEXIST corresponds to FileExistsError
            raise e
    yield path
    shutil.rmtree(path)


def test_invalid_formats(es, path_management):
    error_text = 'must be one of the following formats: {}'
    error_text = error_text.format(', '.join(serialize.FORMATS))
    with pytest.raises(ValueError, match=error_text):
        serialize.write_entity_data(es.entities[0], path=path_management, format='')
    with pytest.raises(ValueError, match=error_text):
        entity = {'loading_info': {'location': 'data', 'type': ''}}
        deserialize.read_entity_data(entity, path='.')


def test_empty_dataframe(es):
    for entity in es.entities:
        description = serialize.entity_to_description(entity)
        dataframe = deserialize.empty_dataframe(description)
        assert dataframe.empty


def test_to_csv(es, path_management):
    es.to_csv(path_management, encoding='utf-8', engine='python')
    new_es = deserialize.read_entityset(path_management)
    assert es.__eq__(new_es, deep=True)


def test_to_pickle(es, path_management):
    es.to_pickle(path_management)
    new_es = deserialize.read_entityset(path_management)
    assert es.__eq__(new_es, deep=True)


def test_to_parquet(es, path_management):
    es.to_parquet(path_management)
    new_es = deserialize.read_entityset(path_management)
    assert es.__eq__(new_es, deep=True)


def test_to_parquet_with_lti(path_management):
    es = load_mock_customer(return_entityset=True, random_seed=0)
    es.to_parquet(path_management)
    new_es = deserialize.read_entityset(path_management)
    assert es.__eq__(new_es, deep=True)


def test_to_pickle_id_none(path_management):
    es = EntitySet()
    es.to_pickle(path_management)
    new_es = deserialize.read_entityset(path_management)
    assert es.__eq__(new_es, deep=True)


def test_serialize_url_csv(es):
    error_text = "Writing to URLs is not supported"
    with pytest.raises(ValueError, match=error_text):
        es.to_csv(URL, encoding='utf-8', engine='python')


def test_deserialize_url_csv(es):
    new_es = deserialize.read_entityset(URL)
    assert es.__eq__(new_es, deep=True)


def test_default_s3_csv(es):
    new_es = deserialize.read_entityset(S3_URL)
    assert es.__eq__(new_es, deep=True)


def test_anon_s3_csv(es):
    new_es = deserialize.read_entityset(S3_URL, profile_name=False)
    assert es.__eq__(new_es, deep=True)


def tests_s3_check_profile(es):
    session = boto3.Session()
    try:
        assert session.get_credentials().access_key is not TEST_KEY
    except AttributeError:
        assert session.get_credentials() is None


@pytest.fixture
def s3_client():
    _environ = dict(os.environ)
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


def test_serialize_s3_csv(es, s3_client, s3_bucket):
    es.to_csv(TEST_S3_URL, encoding='utf-8', engine='python')

    obj = list(s3_bucket.objects.all())[0].key
    s3_client.ObjectAcl(BUCKET_NAME, obj).put(ACL='public-read-write')

    new_es = deserialize.read_entityset(TEST_S3_URL)
    assert es.__eq__(new_es, deep=True)


def test_serialize_s3_pickle(es, s3_client, s3_bucket):
    es.to_pickle(TEST_S3_URL)

    obj = list(s3_bucket.objects.all())[0].key
    s3_client.ObjectAcl(BUCKET_NAME, obj).put(ACL='public-read-write')

    new_es = deserialize.read_entityset(TEST_S3_URL)
    assert es.__eq__(new_es, deep=True)


def test_serialize_s3_parquet(es, s3_client, s3_bucket):
    es.to_parquet(TEST_S3_URL)

    obj = list(s3_bucket.objects.all())[0].key
    s3_client.ObjectAcl(BUCKET_NAME, obj).put(ACL='public-read-write')

    new_es = deserialize.read_entityset(TEST_S3_URL)
    assert es.__eq__(new_es, deep=True)
