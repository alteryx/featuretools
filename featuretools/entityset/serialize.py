import json
import os
import shutil

from .. import variable_types

FORMATS = ['csv', 'pickle', 'parquet']
VARIABLE_TYPES = {
    str(getattr(variable_types, type).type_string): getattr(variable_types, type) for type in dir(variable_types)
    if hasattr(getattr(variable_types, type), 'type_string')
}


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


def relationship_to_description(relationship):
    '''Serialize entityset relationship to data description.

    Args:
        relationship (Relationship) : Instance of :class:`.Relationship`.

    Returns:
        description (dict) : Description of :class:`.Relationship`.
    '''
    return {
        'parent': [relationship.parent_entity.id, relationship.parent_variable.id],
        'child': [relationship.child_entity.id, relationship.child_variable.id],
    }


def entityset_to_description(entityset):
    '''Serialize entityset to data description.

    Args:
        entityset (EntitySet) : Instance of :class:`.EntitySet`.

    Returns:
        description (dict) : Description of :class:`.EntitySet`.
    '''
    entities = {entity.id: entity_to_description(entity) for entity in entityset.entities}
    relationships = [relationship_to_description(relationship) for relationship in entityset.relationships]
    data_description = {
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
