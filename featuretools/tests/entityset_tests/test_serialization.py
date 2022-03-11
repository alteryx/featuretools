import json
import logging
import os
import tempfile
from urllib.request import urlretrieve

import boto3
import pandas as pd
import pytest
import woodwork.type_sys.type_system as ww_type_system
from woodwork.logical_types import Datetime, LogicalType, Ordinal
from woodwork.serializers.serializer_base import typing_info_to_dict
from woodwork.type_sys.utils import list_logical_types

from featuretools.entityset import EntitySet, deserialize, serialize
from featuretools.entityset.serialize import SCHEMA_VERSION
from featuretools.tests.testing_utils import to_pandas
from featuretools.utils.gen_utils import Library

BUCKET_NAME = "test-bucket"
WRITE_KEY_NAME = "test-key"
TEST_S3_URL = "s3://{}/{}".format(BUCKET_NAME, WRITE_KEY_NAME)
TEST_FILE = "test_serialization_data_entityset_schema_{}_2022_2_16.tar".format(SCHEMA_VERSION)
S3_URL = "s3://featuretools-static/" + TEST_FILE
URL = "https://featuretools-static.s3.amazonaws.com/" + TEST_FILE
TEST_KEY = "test_access_key_es"


def test_entityset_description(es):
    description = serialize.entityset_to_description(es)
    _es = deserialize.description_to_entityset(description)
    assert es.metadata.__eq__(_es, deep=True)


def test_all_ww_logical_types():
    logical_types = list_logical_types()['type_string'].to_list()
    dataframe = pd.DataFrame(columns=logical_types)
    es = EntitySet()
    ltype_dict = {ltype: ltype for ltype in logical_types}
    ltype_dict['ordinal'] = Ordinal(order=[])
    es.add_dataframe(dataframe=dataframe, dataframe_name='all_types', index='integer', logical_types=ltype_dict)
    description = serialize.entityset_to_description(es)
    _es = deserialize.description_to_entityset(description)
    assert es.__eq__(_es, deep=True)


def test_with_custom_ww_logical_type():
    class CustomLogicalType(LogicalType):
        pass

    ww_type_system.add_type(CustomLogicalType)
    columns = ['integer', 'natural_language', 'custom_logical_type']
    dataframe = pd.DataFrame(columns=columns)
    es = EntitySet()
    ltype_dict = {
        'integer': 'integer',
        'natural_language': 'natural_language',
        'custom_logical_type': CustomLogicalType,
    }
    es.add_dataframe(dataframe=dataframe, dataframe_name='custom_type', index='integer', logical_types=ltype_dict)
    description = serialize.entityset_to_description(es)
    _es = deserialize.description_to_entityset(description)
    assert isinstance(_es['custom_type'].ww.logical_types['custom_logical_type'], CustomLogicalType)
    assert es.__eq__(_es, deep=True)


def test_serialize_invalid_formats(es, tmpdir):
    error_text = 'must be one of the following formats: {}'
    error_text = error_text.format(', '.join(serialize.FORMATS))
    with pytest.raises(ValueError, match=error_text):
        serialize.write_data_description(es, path=str(tmpdir), format='')


def test_empty_dataframe(es):
    for df in es.dataframes:
        description = typing_info_to_dict(df)
        dataframe = deserialize.empty_dataframe(description)
        assert dataframe.empty
        assert all(dataframe.columns == df.columns)


def test_to_csv(es, tmpdir):
    es.to_csv(str(tmpdir), encoding='utf-8', engine='python')
    new_es = deserialize.read_entityset(str(tmpdir))
    assert es.__eq__(new_es, deep=True)
    df = to_pandas(es['log'], index='id')
    new_df = to_pandas(new_es['log'], index='id')
    assert type(df['latlong'][0]) in (tuple, list)
    assert type(new_df['latlong'][0]) in (tuple, list)


# Dask/Spark don't support auto setting of interesting values with es.add_interesting_values()
def test_to_csv_interesting_values(pd_es, tmpdir):
    pd_es.add_interesting_values()
    pd_es.to_csv(str(tmpdir))
    new_es = deserialize.read_entityset(str(tmpdir))
    assert pd_es.__eq__(new_es, deep=True)


def test_to_csv_manual_interesting_values(es, tmpdir):
    es.add_interesting_values(dataframe_name='log', values={'product_id': ['coke_zero']})
    es.to_csv(str(tmpdir))
    new_es = deserialize.read_entityset(str(tmpdir))
    assert es.__eq__(new_es, deep=True)
    assert new_es['log'].ww['product_id'].ww.metadata['interesting_values'] == ['coke_zero']


# Dask/Spark do not support to_pickle
def test_to_pickle(pd_es, tmpdir):
    pd_es.to_pickle(str(tmpdir))
    new_es = deserialize.read_entityset(str(tmpdir))
    assert pd_es.__eq__(new_es, deep=True)
    assert type(pd_es['log']['latlong'][0]) == tuple
    assert type(new_es['log']['latlong'][0]) == tuple


def test_to_pickle_errors_dask(dask_es, tmpdir):
    msg = 'DataFrame type not compatible with pickle serialization. Please serialize to another format.'
    with pytest.raises(ValueError, match=msg):
        dask_es.to_pickle(str(tmpdir))


def test_to_pickle_errors_spark(spark_es, tmpdir):
    msg = 'DataFrame type not compatible with pickle serialization. Please serialize to another format.'
    with pytest.raises(ValueError, match=msg):
        spark_es.to_pickle(str(tmpdir))


# Dask/Spark do not support to_pickle
def test_to_pickle_interesting_values(pd_es, tmpdir):
    pd_es.add_interesting_values()
    pd_es.to_pickle(str(tmpdir))
    new_es = deserialize.read_entityset(str(tmpdir))
    assert pd_es.__eq__(new_es, deep=True)


# Dask/Spark do not support to_pickle
def test_to_pickle_manual_interesting_values(pd_es, tmpdir):
    pd_es.add_interesting_values(dataframe_name='log', values={'product_id': ['coke_zero']})
    pd_es.to_pickle(str(tmpdir))
    new_es = deserialize.read_entityset(str(tmpdir))
    assert pd_es.__eq__(new_es, deep=True)
    assert new_es['log'].ww['product_id'].ww.metadata['interesting_values'] == ['coke_zero']


def test_to_parquet(es, tmpdir):
    es.to_parquet(str(tmpdir))
    new_es = deserialize.read_entityset(str(tmpdir))
    assert es.__eq__(new_es, deep=True)
    df = to_pandas(es['log'])
    new_df = to_pandas(new_es['log'])
    assert type(df['latlong'][0]) in (tuple, list)
    assert type(new_df['latlong'][0]) in (tuple, list)


def test_to_parquet_manual_interesting_values(es, tmpdir):
    es.add_interesting_values(dataframe_name='log', values={'product_id': ['coke_zero']})
    es.to_parquet(str(tmpdir))
    new_es = deserialize.read_entityset(str(tmpdir))
    assert es.__eq__(new_es, deep=True)
    assert new_es['log'].ww['product_id'].ww.metadata['interesting_values'] == ['coke_zero']


# Dask/Spark don't support auto setting of interesting values with es.add_interesting_values()
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


# TODO: tmp file disappears after deserialize step, cannot check equality with Dask, Spark
@pytest.mark.parametrize("profile_name", [None, False])
def test_serialize_s3_csv(es, s3_client, s3_bucket, profile_name):
    if es.dataframe_type != Library.PANDAS.value:
        pytest.xfail('tmp file disappears after deserialize step, cannot check equality with Dask')
    es.to_csv(TEST_S3_URL, encoding='utf-8', engine='python', profile_name=profile_name)
    make_public(s3_client, s3_bucket)
    new_es = deserialize.read_entityset(TEST_S3_URL, profile_name=profile_name)
    assert es.__eq__(new_es, deep=True)


# Dask and Spark do not support to_pickle
@pytest.mark.parametrize("profile_name", [None, False])
def test_serialize_s3_pickle(pd_es, s3_client, s3_bucket, profile_name):
    pd_es.to_pickle(TEST_S3_URL, profile_name=profile_name)
    make_public(s3_client, s3_bucket)
    new_es = deserialize.read_entityset(TEST_S3_URL, profile_name=profile_name)
    assert pd_es.__eq__(new_es, deep=True)


# TODO: tmp file disappears after deserialize step, cannot check equality with Dask, Spark
@pytest.mark.parametrize("profile_name", [None, False])
def test_serialize_s3_parquet(es, s3_client, s3_bucket, profile_name):
    if es.dataframe_type != Library.PANDAS.value:
        pytest.xfail('tmp file disappears after deserialize step, cannot check equality with Dask or Spark')
    es.to_parquet(TEST_S3_URL, profile_name=profile_name)
    make_public(s3_client, s3_bucket)
    new_es = deserialize.read_entityset(TEST_S3_URL, profile_name=profile_name)
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
    if es.dataframe_type != Library.PANDAS.value:
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
    if es.dataframe_type == Library.SPARK.value:
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


def test_deserialize_s3_csv(es):
    new_es = deserialize.read_entityset(S3_URL, profile_name=False)
    assert es.__eq__(new_es, deep=True)


def test_operations_invalidate_metadata(es):
    new_es = EntitySet(id="test")
    # test metadata gets created on access
    assert new_es._data_description is None
    assert new_es.metadata is not None  # generated after access
    assert new_es._data_description is not None
    if not isinstance(es['customers'], pd.DataFrame):
        customers_ltypes = es["customers"].ww.logical_types
        customers_ltypes['signup_date'] = Datetime
    else:
        customers_ltypes = None
    new_es.add_dataframe(es["customers"],
                         "customers",
                         index=es["customers"].index,
                         logical_types=customers_ltypes)
    if not isinstance(es['sessions'], pd.DataFrame):
        sessions_ltypes = es["sessions"].ww.logical_types
    else:
        sessions_ltypes = None
    new_es.add_dataframe(es["sessions"],
                         "sessions",
                         index=es["sessions"].index,
                         logical_types=sessions_ltypes)

    assert new_es._data_description is None
    assert new_es.metadata is not None
    assert new_es._data_description is not None

    new_es = new_es.add_relationship("customers", "id", "sessions", "customer_id")
    assert new_es._data_description is None
    assert new_es.metadata is not None
    assert new_es._data_description is not None

    new_es = new_es.normalize_dataframe("customers", "cohort", "cohort")
    assert new_es._data_description is None
    assert new_es.metadata is not None
    assert new_es._data_description is not None

    new_es.add_last_time_indexes()
    assert new_es._data_description is None
    assert new_es.metadata is not None
    assert new_es._data_description is not None

    # automatically adding interesting values not supported in Dask or Spark
    if new_es.dataframe_type == Library.PANDAS.value:
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
    dataframes = {dataframe.ww.name: typing_info_to_dict(dataframe) for dataframe in es.dataframes}
    relationships = [relationship.to_dictionary() for relationship in es.relationships]
    dictionary = {
        'schema_version': version,
        'id': es.id,
        'dataframes': dataframes,
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
