import json
import os
import tarfile
import tempfile

import pandas as pd
import woodwork.type_sys.type_system as ww_type_system
from woodwork.deserialize import read_woodwork_table

from featuretools.entityset.relationship import Relationship
from featuretools.utils.gen_utils import Library, import_or_none
from featuretools.utils.s3_utils import get_transport_params, use_smartopen_es
from featuretools.utils.schema_utils import check_schema_version
from featuretools.utils.wrangle import _is_local_tar, _is_s3, _is_url

dd = import_or_none("dask.dataframe")
ps = import_or_none("pyspark.pandas")


def description_to_entityset(description, **kwargs):
    """Deserialize entityset from data description.

    Args:
        description (dict) : Description of an :class:`.EntitySet`. Likely generated using :meth:`.serialize.entityset_to_description`
        kwargs (keywords): Additional keyword arguments to pass as keywords arguments to the underlying deserialization method.

    Returns:
        entityset (EntitySet) : Instance of :class:`.EntitySet`.
    """
    check_schema_version(description, "entityset")

    from featuretools.entityset import EntitySet

    # If data description was not read from disk, path is None.
    path = description.get("path")
    entityset = EntitySet(description["id"])

    for df in description["dataframes"].values():
        if path is not None:
            data_path = os.path.join(path, "data", df["name"])
            format = description.get("format")
            if format is not None:
                kwargs["format"] = format
                if format == "parquet" and df["loading_info"]["table_type"] == "pandas":
                    kwargs["filename"] = df["name"] + ".parquet"
            dataframe = read_woodwork_table(data_path, validate=False, **kwargs)
        else:
            dataframe = empty_dataframe(df, description["data_type"])

        entityset.add_dataframe(dataframe)

    for relationship in description["relationships"]:
        rel = Relationship.from_dictionary(relationship, entityset)
        entityset.add_relationship(relationship=rel)

    return entityset


def empty_dataframe(description, data_type=Library.PANDAS):
    """Deserialize empty dataframe from dataframe description.

    Args:
        description (dict) : Description of dataframe.

    Returns:
        df (DataFrame) : Empty dataframe with Woodwork initialized.
    """
    # TODO: Can we update Woodwork to return an empty initialized dataframe from a description
    # instead of using this function? Or otherwise eliminate? Issue #1476
    logical_types = {}
    semantic_tags = {}
    column_descriptions = {}
    column_metadata = {}
    use_standard_tags = {}
    category_dtypes = {}
    columns = []
    for col in description["column_typing_info"]:
        col_name = col["name"]
        columns.append(col_name)

        ltype_metadata = col["logical_type"]
        ltype = ww_type_system.str_to_logical_type(
            ltype_metadata["type"],
            params=ltype_metadata["parameters"],
        )

        tags = col["semantic_tags"]

        if "index" in tags:
            tags.remove("index")
        elif "time_index" in tags:
            tags.remove("time_index")

        logical_types[col_name] = ltype
        semantic_tags[col_name] = tags
        column_descriptions[col_name] = col["description"]
        column_metadata[col_name] = col["metadata"]
        use_standard_tags[col_name] = col["use_standard_tags"]

        if col["physical_type"]["type"] == "category":
            # Make sure categories are recreated properly
            cat_values = col["physical_type"]["cat_values"]
            cat_dtype = col["physical_type"]["cat_dtype"]
            cat_object = pd.CategoricalDtype(pd.Index(cat_values, dtype=cat_dtype))
            category_dtypes[col_name] = cat_object

    dataframe = pd.DataFrame(columns=columns).astype(category_dtypes)
    if data_type == Library.DASK:
        dataframe = dd.from_pandas(dataframe, npartitions=1)
    elif data_type == Library.SPARK:
        dataframe = ps.from_pandas(dataframe)

    dataframe.ww.init(
        name=description.get("name"),
        index=description.get("index"),
        time_index=description.get("time_index"),
        logical_types=logical_types,
        semantic_tags=semantic_tags,
        use_standard_tags=use_standard_tags,
        table_metadata=description.get("table_metadata"),
        column_metadata=column_metadata,
        column_descriptions=column_descriptions,
        validate=False,
    )

    return dataframe


def read_data_description(path):
    """Read data description from disk, S3 path, or URL.

    Args:
        path (str): Location on disk, S3 path, or URL to read `data_description.json`.

    Returns:
        description (dict) : Description of :class:`.EntitySet`.
    """

    path = os.path.abspath(path)
    assert os.path.exists(path), '"{}" does not exist'.format(path)
    filepath = os.path.join(path, "data_description.json")
    with open(filepath, "r") as file:
        description = json.load(file)
    description["path"] = path
    return description


def read_entityset(path, profile_name=None, **kwargs):
    """Read entityset from disk, S3 path, or URL.

    Args:
        path (str): Directory on disk, S3 path, or URL to read `data_description.json`.
        profile_name (str, bool): The AWS profile specified to write to S3. Will default to None and search for AWS credentials.
            Set to False to use an anonymous profile.
        kwargs (keywords): Additional keyword arguments to pass as keyword arguments to the underlying deserialization method.
    """
    if _is_url(path) or _is_s3(path) or _is_local_tar(str(path)):
        with tempfile.TemporaryDirectory() as tmpdir:
            local_path = path
            transport_params = None

            if _is_s3(path):
                transport_params = get_transport_params(profile_name)

            if _is_s3(path) or _is_url(path):
                local_path = os.path.join(tmpdir, "temporary_es")
                use_smartopen_es(local_path, path, transport_params)

            with tarfile.open(str(local_path)) as tar:
                tar.extractall(path=tmpdir)

            data_description = read_data_description(tmpdir)
            return description_to_entityset(data_description, **kwargs)
    else:
        data_description = read_data_description(path)
        return description_to_entityset(data_description, **kwargs)
