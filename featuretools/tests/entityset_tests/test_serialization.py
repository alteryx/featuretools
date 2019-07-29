import errno
import os
import shutil

import boto3
import pandas as pd
import pytest
from botocore.exceptions import ProfileNotFound
from moto import mock_s3

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
S3_URL = "s3://{}/{}".format(BUCKET_NAME, WRITE_KEY_NAME)
URL = 'https://featuretools-static.s3.amazonaws.com/test_serialization_data_1.0.0.tar'


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


@mock_s3
def test_serialize_s3_csv(es):
    s3 = boto3.resource('s3')
    s3.create_bucket(Bucket=BUCKET_NAME, ACL='public-read-write')

    es.to_csv(S3_URL, encoding='utf-8', engine='python')

    bucket = s3.Bucket(BUCKET_NAME)
    obj = list(bucket.objects.all())[0].key
    s3.ObjectAcl(BUCKET_NAME, obj).put(ACL='public-read-write')

    new_es = deserialize.read_entityset(S3_URL)
    assert es.__eq__(new_es, deep=True)

    for key in boto3.resource('s3').Bucket(BUCKET_NAME).objects.all():
        key.delete()


@mock_s3
def test_serialize_s3_pickle(es):
    s3 = boto3.resource('s3')
    s3.create_bucket(Bucket=BUCKET_NAME, ACL='public-read-write')

    es.to_pickle(S3_URL)

    bucket = s3.Bucket(BUCKET_NAME)
    obj = list(bucket.objects.all())[0].key
    s3.ObjectAcl(BUCKET_NAME, obj).put(ACL='public-read-write')

    new_es = deserialize.read_entityset(S3_URL)
    assert es.__eq__(new_es, deep=True)

    s3 = boto3.resource('s3')
    for key in boto3.resource('s3').Bucket(BUCKET_NAME).objects.all():
        key.delete()
    s3.Bucket(BUCKET_NAME).delete()


@mock_s3
def test_serialize_s3_parquet(es):
    s3 = boto3.resource('s3')
    s3.create_bucket(Bucket=BUCKET_NAME, ACL='public-read-write')

    es.to_parquet(S3_URL)

    bucket = s3.Bucket(BUCKET_NAME)
    obj = list(bucket.objects.all())[0].key
    s3.ObjectAcl(BUCKET_NAME, obj).put(ACL='public-read-write')

    new_es = deserialize.read_entityset(S3_URL)
    assert es.__eq__(new_es, deep=True)

    s3 = boto3.resource('s3')

    for key in boto3.resource('s3').Bucket(BUCKET_NAME).objects.all():
        key.delete()
    s3.Bucket(BUCKET_NAME).delete()


def test_serialize_url_csv(es):
    error_text = "Writing to URLs is not supported"
    with pytest.raises(ValueError, match=error_text):
        es.to_csv(URL, encoding='utf-8', engine='python')


def test_deserialize_url_csv(es):
    new_es = deserialize.read_entityset(URL)
    assert es.__eq__(new_es, deep=True)


def test_real_s3_csv(es):
    test_url = "s3://featuretools-static/test_serialization_data_1.0.0.tar"
    new_es = deserialize.read_entityset(test_url)
    assert es.__eq__(new_es, deep=True)


def tests_s3_profile_serialize(es):
    test_url = "s3://featuretools-static/test_serialization_data_1.0.0.tar"
    error_text = "The config profile (.*) could not be found"
    with pytest.raises(ProfileNotFound, match=error_text):
        es.to_csv(test_url, profile_name="aws")


def tests_s3_profile_deserialize(es):
    test_url = "s3://featuretools-static/test_serialization_data_1.0.0.tar"
    error_text = "The config profile (.*) could not be found"
    with pytest.raises(ProfileNotFound, match=error_text):
        deserialize.read_entityset(test_url, profile_name="aws")
