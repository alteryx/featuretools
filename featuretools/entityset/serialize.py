import datetime
import json
import os
import shutil
import tarfile
import tempfile
import urllib

import boto3
import s3fs
from smart_open import open

FORMATS = ['csv', 'pickle', 'parquet']
SCHEMA_VERSION = "1.0.0"


def entity_to_description(entity):
    '''Serialize entity to data description.

    Args:
        entity (Entity) : Instance of :class:`.Entity`.

    Returns:
        dictionary (dict) : Description of :class:`.Entity`.
    '''
    index = entity.df.columns.isin([variable.id for variable in entity.variables])
    dtypes = entity.df[entity.df.columns[index]].dtypes.astype(str).to_dict()
    description = {
        "id": entity.id,
        "index": entity.index,
        "time_index": entity.time_index,
        "properties": {
            'secondary_time_index': entity.secondary_time_index,
            'last_time_index': entity.last_time_index is not None,
        },
        "variables": [variable.to_data_description() for variable in entity.variables],
        "loading_info": {
            'params': {},
            'properties': {
                'dtypes': dtypes
            }
        }
    }

    return description


def entityset_to_description(entityset):
    '''Serialize entityset to data description.

    Args:
        entityset (EntitySet) : Instance of :class:`.EntitySet`.

    Returns:
        description (dict) : Description of :class:`.EntitySet`.
    '''
    entities = {entity.id: entity_to_description(entity) for entity in entityset.entities}
    relationships = [relationship.to_dictionary() for relationship in entityset.relationships]
    data_description = {
        'schema_version': SCHEMA_VERSION,
        'id': entityset.id,
        'entities': entities,
        'relationships': relationships,
    }
    return data_description


def write_entity_data(entity, path, format='csv', **kwargs):
    '''Write entity data to disk.

    Args:
        entity (Entity) : Instance of :class:`.Entity`.
        path (str) : Location on disk to write entity data.
        format (str) : Format to use for writing entity data. Defaults to csv.
        kwargs (keywords) : Additional keyword arguments to pass as keywords arguments to the underlying serialization method.

    Returns:
        loading_info (dict) : Information on storage location and format of entity data.
    '''
    format = format.lower()
    basename = '.'.join([entity.id, format])
    location = os.path.join('data', basename)
    file = os.path.join(path, location)
    if format == 'csv':
        entity.df.to_csv(
            file,
            index=kwargs['index'],
            sep=kwargs['sep'],
            encoding=kwargs['encoding'],
            compression=kwargs['compression'],
        )
    elif format == 'parquet':
        # Serializing to parquet format raises an error when columns contain tuples.
        # Columns containing tuples are mapped as dtype object.
        # Issue is resolved by casting columns of dtype object to string.
        columns = entity.df.select_dtypes('object').columns
        entity.df[columns] = entity.df[columns].astype('unicode')
        entity.df.columns = entity.df.columns.astype('unicode')  # ensures string column names for python 2.7
        entity.df.to_parquet(file, **kwargs)
    elif format == 'pickle':
        entity.df.to_pickle(file, **kwargs)
    else:
        error = 'must be one of the following formats: {}'
        raise ValueError(error.format(', '.join(FORMATS)))
    return {'location': location, 'type': format, 'params': kwargs}


def write_data_description(entityset, path, **kwargs):
    '''Serialize entityset to data description and write to disk.

    Args:
        entityset (EntitySet) : Instance of :class:`.EntitySet`.
        path (str) : Location on disk to write `data_description.json` and entity data.
        kwargs (keywords) : Additional keyword arguments to pass as keywords arguments to the underlying serialization method.
    '''
    if is_s3(path):
        with tempfile.TemporaryDirectory() as tmpdir:
            os.makedirs(os.path.join(tmpdir, 'data'))
            description = entityset_to_description(entityset)
            for entity in entityset.entities:
                loading_info = write_entity_data(entity, tmpdir, **kwargs)
                description['entities'][entity.id]['loading_info'].update(loading_info)
            file = os.path.join(tmpdir, 'data_description.json')
            with open(file, 'w') as file:
                json.dump(description, file)

            file_name = "es-{date:%Y-%m-%d_%H:%M:%S}".format(date=datetime.datetime.now())
            file_path = os.path.join(tmpdir, file_name)
            tar = tarfile.open(str(file_path) + ".tar", 'w')
            tar.add(str(tmpdir) + '/data_description.json', arcname='data_description.json')
            tar.add(str(tmpdir) + '/data', arcname='/data')
            tar.close()
            tar = tarfile.open(str(file_path) + ".tar")
            if("profile_name" in kwargs):
                transport_params = {'session': boto3.Session(profile_name=kwargs['profile_name'])}
                with open(file_path + ".tar", 'rb') as fin:
                    with open(path, 'wb', transport_params=transport_params) as fout:
                        for line in fin:
                            fout.write(line)
            else:
                s3 = s3fs.S3FileSystem(anon=True)
                with open(file_path + ".tar", 'rb') as fin:
                    with s3.open(path, 'wb') as fout:
                        for line in fin:
                            fout.write(line)
    elif is_url(path):
        raise ValueError("Writing to URLs is not supported")
    else:
        path = os.path.abspath(path)
        if os.path.exists(path):
            shutil.rmtree(path)
        for dirname in [path, os.path.join(path, 'data')]:
            os.makedirs(dirname)
        description = entityset_to_description(entityset)
        for entity in entityset.entities:
            loading_info = write_entity_data(entity, path, **kwargs)
            description['entities'][entity.id]['loading_info'].update(loading_info)
        file = os.path.join(path, 'data_description.json')
        with open(file, 'w') as file:
            json.dump(description, file)


def is_s3(string):
    return urllib.parse.urlparse(string).scheme == 's3'


def is_url(string):
    return urllib.parse.urlparse(string).scheme in ('http', 'https')
