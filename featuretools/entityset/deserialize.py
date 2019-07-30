import json
import os
import tarfile
from pathlib import Path

import boto3
import pandas as pd
import s3fs
from smart_open import open

from featuretools.entityset.relationship import Relationship
from featuretools.entityset.serialize import FORMATS
from featuretools.utils import is_python_2
from featuretools.utils.gen_utils import check_schema_version
from featuretools.utils.wrangle import _is_s3, _is_url
from featuretools.variable_types.variable import find_variable_types

if is_python_2():
    from backports import tempfile
else:
    import tempfile


def description_to_variable(description, entity=None):
    '''Deserialize variable from variable description.

    Args:
        description (dict) : Description of :class:`.Variable`.
        entity (Entity) : Instance of :class:`.Entity` to add :class:`.Variable`. If entity is None, :class:`.Variable` will not be instantiated.

    Returns:
        variable (Variable) : Returns :class:`.Variable`.
    '''
    variable_types = find_variable_types()
    is_type_string = isinstance(description['type'], str)
    type = description['type'] if is_type_string else description['type'].pop('value')
    variable = variable_types.get(type, variable_types.get('None'))  # 'None' will return the Unknown variable type
    if entity is not None:
        kwargs = {} if is_type_string else description['type']
        variable = variable(description['id'], entity, **kwargs)
        variable.interesting_values = description['properties']['interesting_values']
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
    variable_types = {variable['id']: description_to_variable(variable) for variable in description['variables']}
    entityset.entity_from_dataframe(
        description['id'],
        dataframe,
        index=description.get('index'),
        time_index=description.get('time_index'),
        secondary_time_index=description['properties'].get('secondary_time_index'),
        variable_types=variable_types)


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
    if description['loading_info']['type'] == 'csv':
        dataframe = pd.read_csv(
            file,
            engine=kwargs['engine'],
            compression=kwargs['compression'],
            encoding=kwargs['encoding'],
        )
    elif description['loading_info']['type'] == 'parquet':
        dataframe = pd.read_parquet(file, engine=kwargs['engine'])
    elif description['loading_info']['type'] == 'pickle':
        dataframe = pd.read_pickle(file, **kwargs)
    else:
        error = 'must be one of the following formats: {}'
        raise ValueError(error.format(', '.join(FORMATS)))
    dtypes = description['loading_info']['properties']['dtypes']
    return dataframe.astype(dtypes)


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
    with open(file, 'r', encoding='utf-8') as file:
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
            file_name = Path(path).name + ".tar"
            file_path = os.path.join(tmpdir, file_name)
            transport_params = {}
            session = boto3.Session()
            if isinstance(profile_name, str):
                transport_params = {'session': boto3.Session(profile_name=profile_name)}
            if _is_s3(path) and (session.get_credentials() is None or profile_name is False):
                s3 = s3fs.S3FileSystem(anon=True)
                with s3.open(path, "rb") as fin:
                    with open(file_path, 'wb') as fout:
                        for line in fin:
                            fout.write(line)
            else:
                with open(path, "rb", transport_params=transport_params) as fin:
                    with open(file_path, 'wb') as fout:
                        for line in fin:
                            fout.write(line)
            tar = tarfile.open(str(file_path))
            tar.extractall(path=tmpdir)
            data_description = read_data_description(tmpdir)
            return description_to_entityset(data_description, **kwargs)
    else:
        data_description = read_data_description(path)
        return description_to_entityset(data_description, **kwargs)
