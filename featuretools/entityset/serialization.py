import json
import os
import shutil

import pandas as pd

from .. import variable_types

FORMATS = ['csv', 'pickle', 'parquet']
SCHEMA = {
    'entities': ['id', 'index', 'time_index', 'variables', 'properties', 'loading_info'],
    'relationships': {
        'parent': ['parent_entity', 'parent_variable'],
        'child': ['child_entity', 'child_variable']
    }
}
VARIABLE_TYPES = {
    getattr(variable_types, type)._dtype_repr: getattr(variable_types, type)
    for type in dir(variable_types) if hasattr(getattr(variable_types, type), '_dtype_repr')
}


def to_entity_description(entity):
    '''Serialize entity to data description.

    Args:
        entity (Entity) : Instance of :class:`.Entity`.

    Returns:
        dictionary (dict) : Description of :class:`.Entity`.
    '''
    description = {}
    for key in SCHEMA.get('entities'):
        if key in ['id', 'index', 'time_index']:
            description[key] = getattr(entity, key)
        elif key == 'properties':
            description[key] = {
                'secondary_time_index': entity.secondary_time_index,
                'last_time_index': entity.last_time_index is not None,
            }
        elif key == 'variables':
            description[key] = [variable.create_data_description() for variable in entity.variables]
        elif key == 'loading_info':
            description[key] = {'params': {}}
        else:
            raise ValueError('"{}" is not supported'.format(key))
    return description


def to_relationship_description(relationship):
    '''Serialize entityset relationship to data description.

    Args:
        relationship (Relationship) : Instance of :class:`.Relationship`.

    Returns:
        description (dict) : Description of :class:`.Relationship`.
    '''
    return {
        key: [getattr(relationship, attr).id for attr in attrs]
        for key, attrs in SCHEMA['relationships'].items()
    }


def write_entity_data(e, path, type='csv', **kwargs):
    '''Write entity data to disk.

    Args:
        e (Entity) : Instance of :class:`.Entity`.
        path (str) : Location on disk to write entity data.
        type (str) : Format to use for writing entity data. Defaults to csv.
        kwargs (keywords) : Additional keyword arguments to pass as keywords arguments to the underlying serialization method.

    Returns:
        loading_info (dict) : Information on storage location and format of entity data.
    '''
    basename = '.'.join([e.id, type])
    if 'compression' in kwargs:
        basename += '.' + kwargs['compression']
    location = os.path.join('data', basename)
    loading_info = {'location': location, 'type': type.lower()}
    error = 'must be one of the following formats: {}'
    assert loading_info['type'] in FORMATS, error.format(', '.join(FORMATS))
    attr = 'to_{}'.format(loading_info['type'])
    file = os.path.join(path, location)
    obj = e.df.select_dtypes('object').columns
    e.df[obj] = e.df[obj].astype('unicode')
    e.df.columns = e.df.columns.astype('unicode')
    getattr(e.df, attr)(file, **kwargs)
    return loading_info


def write_data_description(es, path, params=None, **kwargs):
    '''Serialize entityset to data description and write to disk.

    Args:
        es (EntitySet) : Instance of :class:`.EntitySet`.
        path (str) : Location on disk to write `data_description.json` and entity data.
        params (dict): Additional keyword arguments to pass as keywords arguments to the underlying deserialization method.
        kwargs (keywords) : Additional keyword arguments to pass as keywords arguments to the underlying serialization method.
    '''
    path = os.path.abspath(path)
    params = params or {}
    if os.path.exists(path):
        shutil.rmtree(path)
    for p in [path, os.path.join(path, 'data')]:
        os.makedirs(p)
    d = es.create_data_description()
    for e in es.entities:
        info = write_entity_data(e, path, **kwargs)
        d['entities'][e.id]['loading_info'].update(info)
        d['entities'][e.id]['loading_info']['params'].update(params)
    file = os.path.join(path, 'data_description.json')
    with open(file, 'w') as f:
        json.dump(d, f)


def from_variable_description(description):
    '''Deserialize variable from data description.

    Args:
        description (dict) : Description of :class:`.Variable`.

    Returns:
        variable (Variable) : Returns type :class:`.Variable`.
    '''
    is_string = isinstance(description['type'], str)
    type = description['type'] if is_string else description['type']['value']
    return VARIABLE_TYPES.get(type, VARIABLE_TYPES.get(None))


def from_entity_description(entityset, entity, path=None):
    '''Deserialize entity from data description and add to entityset.

    Args:
        entityset (EntitySet) : Instance of :class:`.EntitySet` to add :class:`.Entity`.
        entity (dict) : Description of :class:`.Entity`.
        path (str) : Root directory to serialized entityset.
    '''
    entityset.entity_from_dataframe(
        entity['id'],
        read_entity_data(entity, path=path),
        index=entity.get('index'),
        time_index=entity.get('time_index'),
        secondary_time_index=entity['properties'].get('secondary_time_index'),
        variable_types={
            variable['id']: from_variable_description(variable) for variable in entity['variables']
        })


def from_relationship_description(entityset, relationship):
    '''Deserialize parent and child variables from relationship description.

    Args:
        entityset (EntitySet) : Instance of :class:`.EntitySet`.
        relationship (dict) : Description of :class:`.Relationship`.

    Returns:
        item (tuple(Variable, Variable)) : Tuple containing parent and child variables.
    '''
    entity, variable = relationship['parent']
    parent = entityset[entity][variable]
    entity, variable = relationship['child']
    child = entityset[entity][variable]
    return parent, child


def read_entity_data(entity, path=None):
    '''Read entity data from disk.

    Args:
        entity (dict) : Information for loading entity data.

    Returns:
        df (DataFrame) : Instance of dataframe. Returns an empty dataframe path is not specified.
    '''
    if path is None:
        columns = [variable['id'] for variable in entity['variables']]
        return pd.DataFrame(columns=columns)
    file = os.path.join(path, entity['loading_info']['location'])
    params = entity['loading_info'].get('params', {})
    if entity['loading_info']['type'] == 'csv':
        return pd.read_csv(file, **params)
    elif entity['loading_info']['type'] == 'parquet':
        return pd.read_parquet(file, **params)
    elif entity['loading_info']['type'] == 'pickle':
        return pd.read_pickle(file, **params)
    else:
        error = 'must be one of the following formats: {}'
        raise ValueError(error.format(', '.join(FORMATS)))


def read_data_description(path):
    '''Read data description from disk.

        Args:
            path (str): Location on disk to read `data_description.json`.

        Returns:
            d (dict) : Description of :class:`.EntitySet`.
    '''
    path = os.path.abspath(path)
    assert os.path.exists(path), '"{}" does not exist'.format(path)
    file = os.path.join(path, 'data_description.json')
    with open(file, 'r') as f:
        d = json.load(f)
    d['root'] = path
    return d
