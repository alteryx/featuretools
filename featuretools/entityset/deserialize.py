import json
import os
import tarfile
import tempfile
from pathlib import Path

import pandas as pd
from dask import dataframe as dd

from featuretools.entityset.relationship import Relationship
from featuretools.entityset.serialize import FORMATS
from featuretools.utils.gen_utils import check_schema_version
from featuretools.utils.s3_utils import get_transport_params, use_smartopen_es
from featuretools.utils.wrangle import _is_s3, _is_url
from featuretools.variable_types import LatLong, find_variable_types


def description_to_variable(description, entity=None):
    '''Deserialize variable from variable description.

    Args:
        description (dict) : Description of :class:`.Variable`.
        entity (Entity) : Instance of :class:`.Entity` to add :class:`.Variable`. If entity is None, :class:`.Variable` will not be instantiated.

    Returns:
        variable (Variable) : Returns :class:`.Variable`.
    '''
    is_type_string = isinstance(description['type'], str)
    variable = description['type'] if is_type_string else description['type'].pop('value')
    if entity is not None:
        variable_types = find_variable_types()
        variable_class = variable_types.get(variable, variable_types.get('unknown'))
        kwargs = {} if is_type_string else description['type']
        variable = variable_class(description['id'], entity, **kwargs)
        interesting_values = pd.read_json(description['properties']['interesting_values'], typ='series')
        variable.interesting_values = interesting_values
    return variable


def description_to_entity(description, entityset, path=None):
    '''Deserialize entity from entity description and add to entityset.

    Args:
        description (dict) : Description of :class:`.Entity`.
        entityset (EntitySet) : Instance of :class:`.EntitySet` to add :class:`.Entity`.
        path (str) : Root directory to serialized entityset.
    '''
    if path:
        dataframe = read_entity_data(description, path=path)
    else:
        dataframe = empty_dataframe(description)
    variable_types = {variable['id']: (description_to_variable(variable), variable)
                      for variable in description['variables']}
    es = entityset.entity_from_dataframe(
        description['id'],
        dataframe,
        index=description.get('index'),
        time_index=description.get('time_index'),
        secondary_time_index=description['properties'].get('secondary_time_index'),
        variable_types={variable: variable_types[variable][0] for variable in variable_types})
    for variable in es[description['id']].variables:
        interesting_values = variable_types[variable.id][1]['properties']['interesting_values']
        interesting_values = pd.read_json(interesting_values, typ="series")
        variable.interesting_values = interesting_values


def description_to_entityset(description, **kwargs):
    '''Deserialize entityset from data description.

    Args:
        description (dict) : Description of an :class:`.EntitySet`. Likely generated using :meth:`.serialize.entityset_to_description`
        kwargs (keywords): Additional keyword arguments to pass as keywords arguments to the underlying deserialization method.

    Returns:
        entityset (EntitySet) : Instance of :class:`.EntitySet`.
    '''
    check_schema_version(description, 'entityset')

    from featuretools.entityset import EntitySet
    # If data description was not read from disk, path is None.
    path = description.get('path')
    entityset = EntitySet(description['id'])

    last_time_index = []
    for entity in description['entities'].values():
        entity['loading_info']['params'].update(kwargs)
        # If path is None, an empty dataframe will be created for entity.
        description_to_entity(entity, entityset, path=path)
        if entity['properties']['last_time_index']:
            last_time_index.append(entity['id'])

    for relationship in description['relationships']:
        relationship = Relationship.from_dictionary(relationship, entityset)
        entityset.add_relationship(relationship)

    if len(last_time_index):
        entityset.add_last_time_indexes(updated_entities=last_time_index)

    return entityset


def empty_dataframe(description):
    '''Deserialize empty dataframe from entity description.

    Args:
        description (dict) : Description of :class:`.Entity`.

    Returns:
        df (DataFrame) : Empty dataframe for entity.
    '''
    columns = [variable['id'] for variable in description['variables']]
    dtypes = description['loading_info']['properties']['dtypes']
    return pd.DataFrame(columns=columns).astype(dtypes)


def read_entity_data(description, path):
    '''Read description data from disk.

    Args:
        description (dict) : Description of :class:`.Entity`.
        path (str): Location on disk to read entity data.

    Returns:
        df (DataFrame) : Instance of dataframe.
    '''
    file = os.path.join(path, description['loading_info']['location'])
    kwargs = description['loading_info'].get('params', {})
    load_format = description['loading_info']['type']
    entity_type = description['loading_info'].get('entity_type', 'pandas')
    if entity_type == 'dask':
        lib = dd
    else:
        lib = pd
    if load_format == 'csv':
        dataframe = lib.read_csv(
            file,
            engine=kwargs['engine'],
            compression=kwargs['compression'],
            encoding=kwargs['encoding'],
        )
    elif load_format == 'parquet':
        dataframe = lib.read_parquet(file, engine=kwargs['engine'])
    elif load_format == 'pickle':
        dataframe = pd.read_pickle(file, **kwargs)
    else:
        error = 'must be one of the following formats: {}'
        raise ValueError(error.format(', '.join(FORMATS)))
    dtypes = description['loading_info']['properties']['dtypes']
    dataframe = dataframe.astype(dtypes)

    if load_format in ['parquet', 'csv']:
        latlongs = []
        for var_description in description['variables']:
            if var_description['type']['value'] == LatLong.type_string:
                latlongs.append(var_description["id"])

        def parse_latlong(x):
            return tuple(float(y) for y in x[1:-1].split(","))

        for column in latlongs:
            if entity_type == 'dask':
                meta = (column, tuple([float, float]))
                dataframe[column] = dataframe[column].apply(parse_latlong,
                                                            meta=meta)
            else:
                dataframe[column] = dataframe[column].apply(parse_latlong)

    return dataframe


def read_data_description(path):
    '''Read data description from disk, S3 path, or URL.

        Args:
            path (str): Location on disk, S3 path, or URL to read `data_description.json`.

        Returns:
            description (dict) : Description of :class:`.EntitySet`.
    '''

    path = os.path.abspath(path)
    assert os.path.exists(path), '"{}" does not exist'.format(path)
    file = os.path.join(path, 'data_description.json')
    with open(file, 'r') as file:
        description = json.load(file)
    description['path'] = path
    return description


def read_entityset(path, profile_name=None, **kwargs):
    '''Read entityset from disk, S3 path, or URL.

        Args:
            path (str): Directory on disk, S3 path, or URL to read `data_description.json`.
            profile_name (str, bool): The AWS profile specified to write to S3. Will default to None and search for AWS credentials.
                Set to False to use an anonymous profile.
            kwargs (keywords): Additional keyword arguments to pass as keyword arguments to the underlying deserialization method.
    '''
    if _is_url(path) or _is_s3(path):
        with tempfile.TemporaryDirectory() as tmpdir:
            file_name = Path(path).name
            file_path = os.path.join(tmpdir, file_name)
            transport_params = None

            if _is_s3(path):
                transport_params = get_transport_params(profile_name)

            use_smartopen_es(file_path, path, transport_params)
            with tarfile.open(str(file_path)) as tar:
                tar.extractall(path=tmpdir)

            data_description = read_data_description(tmpdir)
            return description_to_entityset(data_description, **kwargs)
    else:
        data_description = read_data_description(path)
        return description_to_entityset(data_description, **kwargs)
