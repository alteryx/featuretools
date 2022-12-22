import datetime
import json
import os
import tarfile
import tempfile

from woodwork.serializers.serializer_base import typing_info_to_dict

from featuretools.utils.gen_utils import import_or_none
from featuretools.utils.s3_utils import get_transport_params, use_smartopen_es
from featuretools.utils.wrangle import _is_s3, _is_url
from featuretools.version import ENTITYSET_SCHEMA_VERSION

ps = import_or_none("pyspark.pandas")

FORMATS = ["csv", "pickle", "parquet"]


def entityset_to_description(entityset, format=None):
    """Serialize entityset to data description.

    Args:
        entityset (EntitySet) : Instance of :class:`.EntitySet`.

    Returns:
        description (dict) : Description of :class:`.EntitySet`.
    """

    dataframes = {
        dataframe.ww.name: typing_info_to_dict(dataframe)
        for dataframe in entityset.dataframes
    }
    relationships = [
        relationship.to_dictionary() for relationship in entityset.relationships
    ]

    data_type = entityset.dataframe_type

    data_description = {
        "schema_version": ENTITYSET_SCHEMA_VERSION,
        "id": entityset.id,
        "dataframes": dataframes,
        "relationships": relationships,
        "format": format,
        "data_type": data_type,
    }
    return data_description


def write_data_description(entityset, path, profile_name=None, **kwargs):
    """Serialize entityset to data description and write to disk or S3 path.

    Args:
        entityset (EntitySet) : Instance of :class:`.EntitySet`.
        path (str) : Location on disk or S3 path to write `data_description.json` and dataframe data.
        profile_name (str, bool): The AWS profile specified to write to S3. Will default to None and search for AWS credentials.
            Set to False to use an anonymous profile.
        kwargs (keywords) : Additional keyword arguments to pass as keywords arguments to the underlying serialization method or to specify AWS profile.
    """
    if _is_s3(path):
        with tempfile.TemporaryDirectory() as tmpdir:
            os.makedirs(os.path.join(tmpdir, "data"))
            dump_data_description(entityset, tmpdir, **kwargs)
            file_path = create_archive(tmpdir)

            transport_params = get_transport_params(profile_name)
            use_smartopen_es(
                file_path,
                path,
                read=False,
                transport_params=transport_params,
            )
    elif _is_url(path):
        raise ValueError("Writing to URLs is not supported")
    else:
        path = os.path.abspath(path)
        os.makedirs(os.path.join(path, "data"), exist_ok=True)
        dump_data_description(entityset, path, **kwargs)


def dump_data_description(entityset, path, **kwargs):
    format = kwargs.get("format")
    description = entityset_to_description(entityset, format)
    for df in entityset.dataframes:
        data_path = os.path.join(path, "data", df.ww.name)
        os.makedirs(os.path.join(data_path, "data"), exist_ok=True)
        df.ww.to_disk(data_path, **kwargs)
    file = os.path.join(path, "data_description.json")
    with open(file, "w") as file:
        json.dump(description, file)


def create_archive(tmpdir):
    file_name = "es-{date:%Y-%m-%d_%H%M%S}.tar".format(date=datetime.datetime.now())
    file_path = os.path.join(tmpdir, file_name)
    tar = tarfile.open(str(file_path), "w")
    tar.add(str(tmpdir) + "/data_description.json", arcname="/data_description.json")
    tar.add(str(tmpdir) + "/data", arcname="/data")
    tar.close()
    return file_path
