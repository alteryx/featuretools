import datetime
import json
import os
import tarfile
import tempfile

import dask.dataframe as dd

from featuretools.utils.s3_utils import get_transport_params, use_smartopen_es
from featuretools.utils.wrangle import _is_s3, _is_url

FORMATS = ['csv', 'pickle', 'parquet']
SCHEMA_VERSION = "4.0.0"


def entity_to_description(entity):
    '''Serialize entity to data description.

    Args:
        entity (Entity) : Instance of :class:`.Entity`.

    Returns:
        dictionary (dict) : Description of :class:`.Entity`.
    '''
    index = entity.df.columns.isin([variable.id for variable in entity.variables])
    dtypes = entity.df[entity.df.columns[index]].dtypes.astype(str).to_dict()
    if isinstance(entity.df, dd.DataFrame):
        entity_type = 'dask'
    else:
        entity_type = 'pandas'
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
            'entity_type': entity_type,
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
    entities = {entity.id: entity_to_description(entity) for entity in
                sorted(entityset.entities, key=lambda entity: entity.id)}
    relationships = [relationship.to_dictionary() for relationship in entityset.relationships]
    data_description = {
        'schema_version': SCHEMA_VERSION,
        'id': entityset.id,
        'entities': entities,
        'relationships': relationships,
    }
    return data_description


def write_entity_data(entity, path, format='csv', **kwargs):
    '''Write entity data to disk or S3 path.

    Args:
        entity (Entity) : Instance of :class:`.Entity`.
        path (str) : Location on disk to write entity data.
        format (str) : Format to use for writing entity data. Defaults to csv.
        kwargs (keywords) : Additional keyword arguments to pass as keywords arguments to the underlying serialization method.

    Returns:
        loading_info (dict) : Information on storage location and format of entity data.
    '''
    format = format.lower()
    if isinstance(entity.df, dd.DataFrame) and format == 'csv':
        basename = "{}-*.{}".format(entity.id, format)
    else:
        basename = '.'.join([entity.id, format])
    location = os.path.join('data', basename)
    file = os.path.join(path, location)
    df = entity.df

    if format == 'csv':
        df.to_csv(
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
        df = df.copy()
        columns = list(df.select_dtypes('object').columns)
        df[columns] = df[columns].astype('unicode')
        df.to_parquet(file, **kwargs)
    elif format == 'pickle':
        # Dask currently does not support to_pickle
        if isinstance(df, dd.DataFrame):
            msg = 'Cannot serialize Dask EntitySet to pickle'
            raise ValueError(msg)
        else:
            df.to_pickle(file, **kwargs)
    else:
        error = 'must be one of the following formats: {}'
        raise ValueError(error.format(', '.join(FORMATS)))
    return {'location': location, 'type': format, 'params': kwargs}


def write_data_description(entityset, path, profile_name=None, **kwargs):
    '''Serialize entityset to data description and write to disk or S3 path.

    Args:
        entityset (EntitySet) : Instance of :class:`.EntitySet`.
        path (str) : Location on disk or S3 path to write `data_description.json` and entity data.
        profile_name (str, bool): The AWS profile specified to write to S3. Will default to None and search for AWS credentials.
            Set to False to use an anonymous profile.
        kwargs (keywords) : Additional keyword arguments to pass as keywords arguments to the underlying serialization method or to specify AWS profile.
    '''
    if _is_s3(path):
        with tempfile.TemporaryDirectory() as tmpdir:
            os.makedirs(os.path.join(tmpdir, 'data'))
            dump_data_description(entityset, tmpdir, **kwargs)
            file_path = create_archive(tmpdir)

            transport_params = get_transport_params(profile_name)
            use_smartopen_es(file_path, path, read=False, transport_params=transport_params)
    elif _is_url(path):
        raise ValueError("Writing to URLs is not supported")
    else:
        path = os.path.abspath(path)
        os.makedirs(os.path.join(path, 'data'), exist_ok=True)
        dump_data_description(entityset, path, **kwargs)


def dump_data_description(entityset, path, **kwargs):
    description = entityset_to_description(entityset)
    for entity in entityset.entities:
        loading_info = write_entity_data(entity, path, **kwargs)
        description['entities'][entity.id]['loading_info'].update(loading_info)
    file = os.path.join(path, 'data_description.json')
    with open(file, 'w') as file:
        json.dump(description, file)


def create_archive(tmpdir):
    file_name = "es-{date:%Y-%m-%d_%H%M%S}.tar".format(date=datetime.datetime.now())
    file_path = os.path.join(tmpdir, file_name)
    tar = tarfile.open(str(file_path), 'w')
    tar.add(str(tmpdir) + '/data_description.json', arcname='/data_description.json')
    tar.add(str(tmpdir) + '/data', arcname='/data')
    tar.close()
    return file_path
