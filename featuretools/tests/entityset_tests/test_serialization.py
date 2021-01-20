import json
import logging
import os
import tempfile
from urllib.request import urlretrieve

import boto3
import pandas as pd
import pytest

from featuretools import variable_types
from featuretools.entityset import (
    EntitySet,
    Relationship,
    deserialize,
    serialize
)
from featuretools.entityset.serialize import SCHEMA_VERSION
from featuretools.tests.testing_utils import to_pandas
from featuretools.utils.gen_utils import import_or_none
from featuretools.variable_types import (
    Categorical,
    Index,
    TimeIndex,
    find_variable_types
)

ks = import_or_none('databricks.koalas')


BUCKET_NAME = "test-bucket"
WRITE_KEY_NAME = "test-key"
TEST_S3_URL = "s3://{}/{}".format(BUCKET_NAME, WRITE_KEY_NAME)
TEST_FILE = "test_serialization_data_entityset_schema_5.1.0.tar"
S3_URL = "s3://featuretools-static/" + TEST_FILE
URL = "https://featuretools-static.s3.amazonaws.com/" + TEST_FILE
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


def test_unknown_variable_description(es):
    description = {'type': 'some_unknown_type', 'id': 'some_unknown_id', 'properties': {'name': 'some_unknown_type', 'interesting_values': '{}'}}
    variable = deserialize.description_to_variable(description, entity=es.entities[0])
    assert(variable.dtype == 'unknown')


def test_custom_description_variable_description(es):
    for entity in es.entities:
        for variable in entity.variables:
            variable.description = 'a custom description'
            description = variable.to_data_description()
            _variable = deserialize.description_to_variable(description, entity=entity)
            assert _variable.description == 'a custom description'


def test_entity_descriptions(es):
    _es = EntitySet(es.id)
    for entity in es.metadata.entities:
        description = serialize.entity_to_description(entity)
        deserialize.description_to_entity(description, _es)
        _entity = _es[description['id']]
        _entity.last_time_index = entity.last_time_index
        assert entity.__eq__(_entity, deep=True)


def test_dask_entity_descriptions(dask_es):
    _es = EntitySet(dask_es.id)
    for entity in dask_es.metadata.entities:
        description = serialize.entity_to_description(entity)
        deserialize.description_to_entity(description, _es)
        _entity = _es[description['id']]
        _entity.last_time_index = entity.last_time_index
        assert entity.__eq__(_entity, deep=True)


def test_entityset_description(es):
    description = serialize.entityset_to_description(es)
    _es = deserialize.description_to_entityset(description)
    assert es.metadata.__eq__(_es, deep=True)


def test_dask_entityset_description(dask_es):
    description = serialize.entityset_to_description(dask_es)
    _es = deserialize.description_to_entityset(description)
    assert dask_es.metadata.__eq__(_es, deep=True)


def test_invalid_formats(es, tmpdir):
    error_text = 'must be one of the following formats: {}'
    error_text = error_text.format(', '.join(serialize.FORMATS))
    with pytest.raises(ValueError, match=error_text):
        serialize.write_entity_data(es.entities[0], path=str(tmpdir), format='')
    with pytest.raises(ValueError, match=error_text):
        entity = {'loading_info': {'location': 'data', 'type': ''}}
        deserialize.read_entity_data(entity, path='.')


def test_empty_dataframe(es):
    for entity in es.entities:
        description = serialize.entity_to_description(entity)
        dataframe = deserialize.empty_dataframe(description)
        assert dataframe.empty


def test_to_csv(es, tmpdir):
    es.to_csv(str(tmpdir), encoding='utf-8', engine='python')
    new_es = deserialize.read_entityset(str(tmpdir))
    assert es.__eq__(new_es, deep=True)
    df = to_pandas(es['log'].df, index='id')
    new_df = to_pandas(new_es['log'].df, index='id')
    assert type(df['latlong'][0]) in (tuple, list)
    assert type(new_df['latlong'][0]) in (tuple, list)


# Dask does not support to_pickle
def test_to_pickle(pd_es, tmpdir):
    pd_es.to_pickle(str(tmpdir))
    new_es = deserialize.read_entityset(str(tmpdir))
    assert pd_es.__eq__(new_es, deep=True)
    assert type(pd_es['log'].df['latlong'][0]) == tuple
    assert type(new_es['log'].df['latlong'][0]) == tuple


def test_to_pickle_errors_dask(dask_es, tmpdir):
    msg = 'Cannot serialize Dask EntitySet to pickle'
    with pytest.raises(ValueError, match=msg):
        dask_es.to_pickle(str(tmpdir))


# Dask does not support to_pickle
def test_to_pickle_interesting_values(pd_es, tmpdir):
    pd_es.add_interesting_values()
    pd_es.to_pickle(str(tmpdir))
    new_es = deserialize.read_entityset(str(tmpdir))
    assert pd_es.__eq__(new_es, deep=True)


# Dask does not support to_pickle
def test_to_pickle_manual_interesting_values(pd_es, tmpdir):
    pd_es['log']['product_id'].interesting_values = ["coke_zero"]
    pd_es.to_pickle(str(tmpdir))
    new_es = deserialize.read_entityset(str(tmpdir))
    assert pd_es.__eq__(new_es, deep=True)


def test_to_parquet(es, tmpdir):
    es.to_parquet(str(tmpdir))
    new_es = deserialize.read_entityset(str(tmpdir))
    assert es.__eq__(new_es, deep=True)
    df = to_pandas(es['log'].df)
    new_df = to_pandas(new_es['log'].df)
    assert type(df['latlong'][0]) in (tuple, list)
    assert type(new_df['latlong'][0]) in (tuple, list)


def test_to_parquet_manual_interesting_values(es, tmpdir):
    es['log']['product_id'].interesting_values = ["coke_zero"]
    es.to_parquet(str(tmpdir))
    new_es = deserialize.read_entityset(str(tmpdir))
    assert es.__eq__(new_es, deep=True)


# Dask does not support es.add_interesting_values
def test_dask_to_parquet_manual_interesting_values(dask_es, tmpdir):
    dask_es['log']['product_id'].interesting_values = ["coke_zero"]
    dask_es.to_parquet(str(tmpdir))
    new_es = deserialize.read_entityset(str(tmpdir))
    assert dask_es.__eq__(new_es, deep=True)


# Dask doesn't support es.add_interesting_values
def test_to_parquet_interesting_values(pd_es, tmpdir):
    pd_es.add_interesting_values()
    pd_es.to_parquet(str(tmpdir))
    new_es = deserialize.read_entityset(str(tmpdir))
    assert pd_es.__eq__(new_es, deep=True)


def test_to_parquet_with_lti(tmpdir, pd_mock_customer):
    es = pd_mock_customer
    es.to_parquet(str(tmpdir))
    new_es = deserialize.read_entityset(str(tmpdir))
    assert es.__eq__(new_es, deep=True)


def test_to_pickle_id_none(tmpdir):
    es = EntitySet()
    es.to_pickle(str(tmpdir))
    new_es = deserialize.read_entityset(str(tmpdir))
    assert es.__eq__(new_es, deep=True)


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


# TODO: tmp file disappears after deserialize step, cannot check equality with Dask
def test_serialize_s3_csv(es, s3_client, s3_bucket):
    if not all(isinstance(entity.df, pd.DataFrame) for entity in es.entities):
        pytest.xfail('tmp file disappears after deserialize step, cannot check equality with Dask')
    es.to_csv(TEST_S3_URL, encoding='utf-8', engine='python')
    make_public(s3_client, s3_bucket)
    new_es = deserialize.read_entityset(TEST_S3_URL)
    assert es.__eq__(new_es, deep=True)


# Dask and Koalas do not support to_pickle
def test_serialize_s3_pickle(pd_es, s3_client, s3_bucket):
    pd_es.to_pickle(TEST_S3_URL)
    make_public(s3_client, s3_bucket)
    new_es = deserialize.read_entityset(TEST_S3_URL)
    assert pd_es.__eq__(new_es, deep=True)


# TODO: tmp file disappears after deserialize step, cannot check equality with Dask, Koalas
def test_serialize_s3_parquet(es, s3_client, s3_bucket):
    if not all(isinstance(entity.df, pd.DataFrame) for entity in es.entities):
        pytest.xfail('tmp file disappears after deserialize step, cannot check equality with Dask or Koalas')
    es.to_parquet(TEST_S3_URL)
    make_public(s3_client, s3_bucket)
    new_es = deserialize.read_entityset(TEST_S3_URL)
    assert es.__eq__(new_es, deep=True)


# TODO: tmp file disappears after deserialize step, cannot check equality with Dask, Koalas
def test_serialize_s3_anon_csv(es, s3_client, s3_bucket):
    if not all(isinstance(entity.df, pd.DataFrame) for entity in es.entities):
        pytest.xfail('tmp file disappears after deserialize step, cannot check equality with Dask or Koalas')
    es.to_csv(TEST_S3_URL, encoding='utf-8', engine='python', profile_name=False)
    make_public(s3_client, s3_bucket)
    new_es = deserialize.read_entityset(TEST_S3_URL, profile_name=False)
    assert es.__eq__(new_es, deep=True)


# Dask does not support to_pickle
def test_serialize_s3_anon_pickle(pd_es, s3_client, s3_bucket):
    pd_es.to_pickle(TEST_S3_URL, profile_name=False)
    make_public(s3_client, s3_bucket)
    new_es = deserialize.read_entityset(TEST_S3_URL, profile_name=False)
    assert pd_es.__eq__(new_es, deep=True)


# TODO: tmp file disappears after deserialize step, cannot check equality with Dask
def test_serialize_s3_anon_parquet(es, s3_client, s3_bucket):
    if not all(isinstance(entity.df, pd.DataFrame) for entity in es.entities):
        pytest.xfail('tmp file disappears after deserialize step, cannot check equality with Dask')
    es.to_parquet(TEST_S3_URL, profile_name=False)
    make_public(s3_client, s3_bucket)
    new_es = deserialize.read_entityset(TEST_S3_URL, profile_name=False)
    assert es.__eq__(new_es, deep=True)


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


def test_s3_test_profile(es, s3_client, s3_bucket, setup_test_profile):
    if not all(isinstance(entity.df, pd.DataFrame) for entity in es.entities):
        pytest.xfail('tmp file disappears after deserialize step, cannot check equality with Dask')
    es.to_csv(TEST_S3_URL, encoding='utf-8', engine='python', profile_name='test')
    make_public(s3_client, s3_bucket)
    new_es = deserialize.read_entityset(TEST_S3_URL, profile_name='test')
    assert es.__eq__(new_es, deep=True)


def test_serialize_url_csv(es):
    error_text = "Writing to URLs is not supported"
    with pytest.raises(ValueError, match=error_text):
        es.to_csv(URL, encoding='utf-8', engine='python')


def test_serialize_subdirs_not_removed(es, tmpdir):
    write_path = tmpdir.mkdir("test")
    test_dir = write_path.mkdir("test_dir")
    with open(str(write_path.join('data_description.json')), 'w') as f:
        json.dump('__SAMPLE_TEXT__', f)
    if ks and any(isinstance(e.df, ks.DataFrame) for e in es.entities):
        compression = 'none'
    else:
        compression = None
    serialize.write_data_description(es, path=str(write_path), index='1', sep='\t', encoding='utf-8', compression=compression)
    assert os.path.exists(str(test_dir))
    with open(str(write_path.join('data_description.json')), 'r') as f:
        assert '__SAMPLE_TEXT__' not in json.load(f)


def test_deserialize_local_tar(es):
    with tempfile.TemporaryDirectory() as tmpdir:
        temp_tar_filepath = os.path.join(tmpdir, TEST_FILE)
        urlretrieve(URL, filename=temp_tar_filepath)
        new_es = deserialize.read_entityset(temp_tar_filepath)
        assert es.__eq__(new_es, deep=True)


def test_deserialize_url_csv(es):
    new_es = deserialize.read_entityset(URL)
    assert es.__eq__(new_es, deep=True)


def test_default_s3_csv(es):
    new_es = deserialize.read_entityset(S3_URL)
    assert es.__eq__(new_es, deep=True)


def test_anon_s3_csv(es):
    new_es = deserialize.read_entityset(S3_URL, profile_name=False)
    assert es.__eq__(new_es, deep=True)


def test_operations_invalidate_metadata(es):
    new_es = EntitySet(id="test")
    # test metadata gets created on access
    assert new_es._data_description is None
    assert new_es.metadata is not None  # generated after access
    assert new_es._data_description is not None
    if not isinstance(es['customers'].df, pd.DataFrame):
        customers_vtypes = es["customers"].variable_types
        customers_vtypes['signup_date'] = variable_types.Datetime
    else:
        customers_vtypes = None
    new_es.entity_from_dataframe("customers",
                                 es["customers"].df,
                                 index=es["customers"].index,
                                 variable_types=customers_vtypes)
    if not isinstance(es['sessions'].df, pd.DataFrame):
        sessions_vtypes = es["sessions"].variable_types
    else:
        sessions_vtypes = None
    new_es.entity_from_dataframe("sessions",
                                 es["sessions"].df,
                                 index=es["sessions"].index,
                                 variable_types=sessions_vtypes)
    assert new_es._data_description is None
    assert new_es.metadata is not None
    assert new_es._data_description is not None

    r = Relationship(new_es["customers"]["id"],
                     new_es["sessions"]["customer_id"])
    new_es = new_es.add_relationship(r)
    assert new_es._data_description is None
    assert new_es.metadata is not None
    assert new_es._data_description is not None

    new_es = new_es.normalize_entity("customers", "cohort", "cohort")
    assert new_es._data_description is None
    assert new_es.metadata is not None
    assert new_es._data_description is not None

    new_es.add_last_time_indexes()
    assert new_es._data_description is None
    assert new_es.metadata is not None
    assert new_es._data_description is not None

    # automatically adding interesting values not supported in Dask or Koalas
    if any(isinstance(entity.df, pd.DataFrame) for entity in new_es.entities):
        new_es.add_interesting_values()
        assert new_es._data_description is None
        assert new_es.metadata is not None
        assert new_es._data_description is not None


def test_reset_metadata(es):
    assert es.metadata is not None
    assert es._data_description is not None
    es.reset_data_description()
    assert es._data_description is None


def test_later_schema_version(es, caplog):
    def test_version(major, minor, patch, raises=True):
        version = '.'.join([str(v) for v in [major, minor, patch]])
        if raises:
            warning_text = ('The schema version of the saved entityset'
                            '(%s) is greater than the latest supported (%s). '
                            'You may need to upgrade featuretools. Attempting to load entityset ...'
                            % (version, SCHEMA_VERSION))
        else:
            warning_text = None

        _check_schema_version(version, es, warning_text, caplog, 'warn')

    major, minor, patch = [int(s) for s in SCHEMA_VERSION.split('.')]

    test_version(major + 1, minor, patch)
    test_version(major, minor + 1, patch)
    test_version(major, minor, patch + 1)
    test_version(major, minor - 1, patch + 1, raises=False)


def test_earlier_schema_version(es, caplog):
    def test_version(major, minor, patch, raises=True):
        version = '.'.join([str(v) for v in [major, minor, patch]])
        if raises:
            warning_text = ('The schema version of the saved entityset'
                            '(%s) is no longer supported by this version '
                            'of featuretools. Attempting to load entityset ...'
                            % (version))
        else:
            warning_text = None

        _check_schema_version(version, es, warning_text, caplog, 'log')

    major, minor, patch = [int(s) for s in SCHEMA_VERSION.split('.')]

    test_version(major - 1, minor, patch)
    test_version(major, minor - 1, patch, raises=False)
    test_version(major, minor, patch - 1, raises=False)


def _check_schema_version(version, es, warning_text, caplog, warning_type=None):
    entities = {entity.id: serialize.entity_to_description(entity) for entity in es.entities}
    relationships = [relationship.to_dictionary() for relationship in es.relationships]
    dictionary = {
        'schema_version': version,
        'id': es.id,
        'entities': entities,
        'relationships': relationships,
    }

    if warning_type == 'log' and warning_text:
        logger = logging.getLogger('featuretools')
        logger.propagate = True
        deserialize.description_to_entityset(dictionary)
        assert warning_text in caplog.text
        logger.propagate = False
    elif warning_type == 'warn' and warning_text:
        with pytest.warns(UserWarning) as record:
            deserialize.description_to_entityset(dictionary)
        assert record[0].message.args[0] == warning_text
    else:
        deserialize.description_to_entityset(dictionary)
