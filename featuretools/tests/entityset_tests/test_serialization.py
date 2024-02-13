import json
import logging
import os
import tempfile
from unittest.mock import patch
from urllib.request import urlretrieve

import boto3
import pandas as pd
import pytest
import woodwork.type_sys.type_system as ww_type_system
from woodwork.logical_types import Datetime, LogicalType, Ordinal
from woodwork.serializers.serializer_base import typing_info_to_dict
from woodwork.type_sys.utils import list_logical_types

from featuretools.entityset import EntitySet, deserialize, serialize
from featuretools.tests.testing_utils import to_pandas
from featuretools.utils.gen_utils import Library
from featuretools.version import ENTITYSET_SCHEMA_VERSION

BUCKET_NAME = "test-bucket"
WRITE_KEY_NAME = "test-key"
TEST_S3_URL = "s3://{}/{}".format(BUCKET_NAME, WRITE_KEY_NAME)
TEST_FILE = "test_serialization_data_entityset_schema_{}_2022_09_02.tar".format(
    ENTITYSET_SCHEMA_VERSION,
)
S3_URL = "s3://featuretools-static/" + TEST_FILE
URL = "https://featuretools-static.s3.amazonaws.com/" + TEST_FILE
TEST_KEY = "test_access_key_es"


def test_entityset_description(es):
    description = serialize.entityset_to_description(es)
    _es = deserialize.description_to_entityset(description)
    assert es.metadata.__eq__(_es, deep=True)


def test_all_ww_logical_types():
    logical_types = list_logical_types()["type_string"].to_list()
    dataframe = pd.DataFrame(columns=logical_types)
    es = EntitySet()
    ltype_dict = {ltype: ltype for ltype in logical_types}
    ltype_dict["ordinal"] = Ordinal(order=[])
    es.add_dataframe(
        dataframe=dataframe,
        dataframe_name="all_types",
        index="integer",
        logical_types=ltype_dict,
    )
    description = serialize.entityset_to_description(es)
    _es = deserialize.description_to_entityset(description)
    assert es.__eq__(_es, deep=True)


def test_with_custom_ww_logical_type():
    class CustomLogicalType(LogicalType):
        pass

    ww_type_system.add_type(CustomLogicalType)
    columns = ["integer", "natural_language", "custom_logical_type"]
    dataframe = pd.DataFrame(columns=columns)
    es = EntitySet()
    ltype_dict = {
        "integer": "integer",
        "natural_language": "natural_language",
        "custom_logical_type": CustomLogicalType,
    }
    es.add_dataframe(
        dataframe=dataframe,
        dataframe_name="custom_type",
        index="integer",
        logical_types=ltype_dict,
    )
    description = serialize.entityset_to_description(es)
    _es = deserialize.description_to_entityset(description)
    assert isinstance(
        _es["custom_type"].ww.logical_types["custom_logical_type"],
        CustomLogicalType,
    )
    assert es.__eq__(_es, deep=True)


def test_serialize_invalid_formats(es, tmp_path):
    error_text = "must be one of the following formats: {}"
    error_text = error_text.format(", ".join(serialize.FORMATS))
    with pytest.raises(ValueError, match=error_text):
        serialize.write_data_description(es, path=str(tmp_path), format="")


def test_empty_dataframe(es):
    for df in es.dataframes:
        description = typing_info_to_dict(df)
        dataframe = deserialize.empty_dataframe(description)
        assert dataframe.empty
        assert all(dataframe.columns == df.columns)


def test_to_csv(es, tmp_path):
    es.to_csv(str(tmp_path), encoding="utf-8", engine="python")
    new_es = deserialize.read_entityset(str(tmp_path))
    assert es.__eq__(new_es, deep=True)
    df = to_pandas(es["log"], index="id")
    new_df = to_pandas(new_es["log"], index="id")
    assert type(df["latlong"][0]) in (tuple, list)
    assert type(new_df["latlong"][0]) in (tuple, list)


# Dask/Spark don't support auto setting of interesting values with es.add_interesting_values()
def test_to_csv_interesting_values(pd_es, tmp_path):
    pd_es.add_interesting_values()
    pd_es.to_csv(str(tmp_path))
    new_es = deserialize.read_entityset(str(tmp_path))
    assert pd_es.__eq__(new_es, deep=True)


def test_to_csv_manual_interesting_values(es, tmp_path):
    es.add_interesting_values(
        dataframe_name="log",
        values={"product_id": ["coke_zero"]},
    )
    es.to_csv(str(tmp_path))
    new_es = deserialize.read_entityset(str(tmp_path))
    assert es.__eq__(new_es, deep=True)
    assert new_es["log"].ww["product_id"].ww.metadata["interesting_values"] == [
        "coke_zero",
    ]


# Dask/Spark do not support to_pickle
def test_to_pickle(pd_es, tmp_path):
    pd_es.to_pickle(str(tmp_path))
    new_es = deserialize.read_entityset(str(tmp_path))
    assert pd_es.__eq__(new_es, deep=True)
    assert type(pd_es["log"]["latlong"][0]) == tuple
    assert type(new_es["log"]["latlong"][0]) == tuple


def test_to_pickle_errors_dask(dask_es, tmp_path):
    msg = "DataFrame type not compatible with pickle serialization. Please serialize to another format."
    with pytest.raises(ValueError, match=msg):
        dask_es.to_pickle(str(tmp_path))


def test_to_pickle_errors_spark(spark_es, tmp_path):
    msg = "DataFrame type not compatible with pickle serialization. Please serialize to another format."
    with pytest.raises(ValueError, match=msg):
        spark_es.to_pickle(str(tmp_path))


# Dask/Spark do not support to_pickle
def test_to_pickle_interesting_values(pd_es, tmp_path):
    pd_es.add_interesting_values()
    pd_es.to_pickle(str(tmp_path))
    new_es = deserialize.read_entityset(str(tmp_path))
    assert pd_es.__eq__(new_es, deep=True)


# Dask/Spark do not support to_pickle
def test_to_pickle_manual_interesting_values(pd_es, tmp_path):
    pd_es.add_interesting_values(
        dataframe_name="log",
        values={"product_id": ["coke_zero"]},
    )
    pd_es.to_pickle(str(tmp_path))
    new_es = deserialize.read_entityset(str(tmp_path))
    assert pd_es.__eq__(new_es, deep=True)
    assert new_es["log"].ww["product_id"].ww.metadata["interesting_values"] == [
        "coke_zero",
    ]


def test_to_parquet(es, tmp_path):
    es.to_parquet(str(tmp_path))
    new_es = deserialize.read_entityset(str(tmp_path))
    assert es.__eq__(new_es, deep=True)
    df = to_pandas(es["log"])
    new_df = to_pandas(new_es["log"])
    assert type(df["latlong"][0]) in (tuple, list)
    assert type(new_df["latlong"][0]) in (tuple, list)


def test_to_parquet_manual_interesting_values(es, tmp_path):
    es.add_interesting_values(
        dataframe_name="log",
        values={"product_id": ["coke_zero"]},
    )
    es.to_parquet(str(tmp_path))
    new_es = deserialize.read_entityset(str(tmp_path))
    assert es.__eq__(new_es, deep=True)
    assert new_es["log"].ww["product_id"].ww.metadata["interesting_values"] == [
        "coke_zero",
    ]


# Dask/Spark don't support auto setting of interesting values with es.add_interesting_values()
def test_to_parquet_interesting_values(pd_es, tmp_path):
    pd_es.add_interesting_values()
    pd_es.to_parquet(str(tmp_path))
    new_es = deserialize.read_entityset(str(tmp_path))
    assert pd_es.__eq__(new_es, deep=True)


def test_to_parquet_with_lti(tmp_path, pd_mock_customer):
    es = pd_mock_customer
    es.to_parquet(str(tmp_path))
    new_es = deserialize.read_entityset(str(tmp_path))
    assert es.__eq__(new_es, deep=True)


def test_to_pickle_id_none(tmp_path):
    es = EntitySet()
    es.to_pickle(str(tmp_path))
    new_es = deserialize.read_entityset(str(tmp_path))
    assert es.__eq__(new_es, deep=True)


# TODO: Fix Moto tests needing to explicitly set permissions for objects
@pytest.fixture
def s3_client():
    _environ = os.environ.copy()
    from moto import mock_aws

    with mock_aws():
        s3 = boto3.resource("s3")
        yield s3
    os.environ.clear()
    os.environ.update(_environ)


@pytest.fixture
def s3_bucket(s3_client, region="us-east-2"):
    location = {"LocationConstraint": region}
    s3_client.create_bucket(
        Bucket=BUCKET_NAME,
        ACL="public-read-write",
        CreateBucketConfiguration=location,
    )
    s3_bucket = s3_client.Bucket(BUCKET_NAME)
    yield s3_bucket


def make_public(s3_client, s3_bucket):
    obj = list(s3_bucket.objects.all())[0].key
    s3_client.ObjectAcl(BUCKET_NAME, obj).put(ACL="public-read-write")


# TODO: tmp file disappears after deserialize step, cannot check equality with Dask, Spark
@pytest.mark.parametrize("profile_name", [None, False])
def test_serialize_s3_csv(es, s3_client, s3_bucket, profile_name):
    if es.dataframe_type != Library.PANDAS:
        pytest.xfail(
            "tmp file disappears after deserialize step, cannot check equality with Dask",
        )
    es.to_csv(TEST_S3_URL, encoding="utf-8", engine="python", profile_name=profile_name)
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
    if es.dataframe_type != Library.PANDAS:
        pytest.xfail(
            "tmp file disappears after deserialize step, cannot check equality with Dask or Spark",
        )
    es.to_parquet(TEST_S3_URL, profile_name=profile_name)
    make_public(s3_client, s3_bucket)
    new_es = deserialize.read_entityset(TEST_S3_URL, profile_name=profile_name)
    assert es.__eq__(new_es, deep=True)


def test_s3_test_profile(es, s3_client, s3_bucket, setup_test_profile):
    if es.dataframe_type != Library.PANDAS:
        pytest.xfail(
            "tmp file disappears after deserialize step, cannot check equality with Dask",
        )
    es.to_csv(TEST_S3_URL, encoding="utf-8", engine="python", profile_name="test")
    make_public(s3_client, s3_bucket)
    new_es = deserialize.read_entityset(TEST_S3_URL, profile_name="test")
    assert es.__eq__(new_es, deep=True)


def test_serialize_url_csv(es):
    error_text = "Writing to URLs is not supported"
    with pytest.raises(ValueError, match=error_text):
        es.to_csv(URL, encoding="utf-8", engine="python")


def test_serialize_subdirs_not_removed(es, tmp_path):
    write_path = tmp_path.joinpath("test")
    write_path.mkdir()
    test_dir = write_path.joinpath("test_dir")
    test_dir.mkdir()
    description_path = write_path.joinpath("data_description.json")
    with open(description_path, "w") as f:
        json.dump("__SAMPLE_TEXT__", f)
    if es.dataframe_type == Library.SPARK:
        compression = "none"
    else:
        compression = None
    serialize.write_data_description(
        es,
        path=str(write_path),
        index="1",
        sep="\t",
        encoding="utf-8",
        compression=compression,
    )
    assert os.path.exists(str(test_dir))
    with open(description_path, "r") as f:
        assert "__SAMPLE_TEXT__" not in json.load(f)


def test_deserialize_local_tar(es):
    with tempfile.TemporaryDirectory() as tmp_path:
        temp_tar_filepath = os.path.join(tmp_path, TEST_FILE)
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
    if not isinstance(es["customers"], pd.DataFrame):
        customers_ltypes = es["customers"].ww.logical_types
        customers_ltypes["signup_date"] = Datetime
    else:
        customers_ltypes = None
    new_es.add_dataframe(
        es["customers"],
        "customers",
        logical_types=customers_ltypes,
    )
    if not isinstance(es["sessions"], pd.DataFrame):
        sessions_ltypes = es["sessions"].ww.logical_types
    else:
        sessions_ltypes = None
    new_es.add_dataframe(
        es["sessions"],
        "sessions",
        logical_types=sessions_ltypes,
    )

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
    if new_es.dataframe_type == Library.PANDAS:
        new_es.add_interesting_values()
        assert new_es._data_description is None
        assert new_es.metadata is not None
        assert new_es._data_description is not None


def test_reset_metadata(es):
    assert es.metadata is not None
    assert es._data_description is not None
    es.reset_data_description()
    assert es._data_description is None


@patch("featuretools.utils.schema_utils.ENTITYSET_SCHEMA_VERSION", "1.1.1")
@pytest.mark.parametrize(
    "hardcoded_schema_version, warns",
    [("2.1.1", True), ("1.2.1", True), ("1.1.2", True), ("1.0.2", False)],
)
def test_later_schema_version(es, caplog, hardcoded_schema_version, warns):
    def test_version(version, warns):
        if warns:
            warning_text = (
                "The schema version of the saved entityset"
                "(%s) is greater than the latest supported (%s). "
                "You may need to upgrade featuretools. Attempting to load entityset ..."
                % (version, "1.1.1")
            )
        else:
            warning_text = None

        _check_schema_version(version, es, warning_text, caplog, "warn")

    test_version(hardcoded_schema_version, warns)


@patch("featuretools.utils.schema_utils.ENTITYSET_SCHEMA_VERSION", "1.1.1")
@pytest.mark.parametrize(
    "hardcoded_schema_version, warns",
    [("0.1.1", True), ("1.0.1", False), ("1.1.0", False)],
)
def test_earlier_schema_version(
    es,
    caplog,
    monkeypatch,
    hardcoded_schema_version,
    warns,
):
    def test_version(version, warns):
        if warns:
            warning_text = (
                "The schema version of the saved entityset"
                "(%s) is no longer supported by this version "
                "of featuretools. Attempting to load entityset ..." % version
            )
        else:
            warning_text = None

        _check_schema_version(version, es, warning_text, caplog, "log")

    test_version(hardcoded_schema_version, warns)


def _check_schema_version(version, es, warning_text, caplog, warning_type=None):
    dataframes = {
        dataframe.ww.name: typing_info_to_dict(dataframe) for dataframe in es.dataframes
    }
    relationships = [relationship.to_dictionary() for relationship in es.relationships]
    dictionary = {
        "schema_version": version,
        "id": es.id,
        "dataframes": dataframes,
        "relationships": relationships,
        "data_type": es.dataframe_type,
    }

    if warning_type == "warn" and warning_text:
        with pytest.warns(UserWarning) as record:
            deserialize.description_to_entityset(dictionary)
        assert record[0].message.args[0] == warning_text
    elif warning_type == "log":
        logger = logging.getLogger("featuretools")
        logger.propagate = True
        deserialize.description_to_entityset(dictionary)
        if warning_text:
            assert warning_text in caplog.text
        else:
            assert not len(caplog.text)
        logger.propagate = False
