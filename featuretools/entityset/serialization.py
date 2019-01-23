import json
import os
import shutil

import pandas as pd

from .. import variable_types

TYPES = ['csv', 'pickle', 'parquet']
SCHEMA = {
    'entities':
    ['id', 'index', 'time_index', 'variables', 'properties', 'loading_info'],
    'relationships': {
        'parent': ['parent_entity', 'parent_variable'],
        'child': ['child_entity', 'child_variable']
    }
}
VTYPES = {
    getattr(variable_types, v)._dtype_repr: getattr(variable_types, v)
    for v in dir(variable_types)
    if hasattr(getattr(variable_types, v), '_dtype_repr')
}


def to_entity_descr(e):
    '''Serialize entity to data description.

    Args:
        e (Entity) : Instance of :class:`.Entity`.

    Returns:
        item (tuple(str, dict)) : Tuple containing id (str) and description (dict) of :class:`.Entity`.
    '''
    descr = {}
    for key in SCHEMA.get('entities'):
        if key in ['id', 'index', 'time_index']:
            descr[key] = getattr(e, key)
        elif key == 'properties':
            sti, lti = e.secondary_time_index, e.last_time_index is not None
            descr[key] = dict(secondary_time_index=sti, last_time_index=lti)
        elif key == 'variables':
            descr[key] = [v.create_data_description() for v in e.variables]
        elif key == 'loading_info':
            dtypes = dict(dtypes=e.df.dtypes.astype(str).to_dict())
            descr[key] = dict(properties=dtypes, params={})
        else:
            raise ValueError('"{}" is not supported'.format(key))
    return e.id, descr


def to_relation_descr(r):
    '''Serialize entityset relationship to data description.

    Args:
        r (Relationship) : Instance of :class:`.Relationship`.

    Returns:
        description (dict) : Description of :class:`.Relationship`.
    '''
    return {
        key: [getattr(r, attr).id for attr in attrs]
        for key, attrs in SCHEMA.get('relationships').items()
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
    loading_info = dict(location=location, type=type.lower())
    assert loading_info['type'] in TYPES, '"{}" is not supported'.format(type)
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


def from_var_descr(d):
    '''Deserialize variable from data description.

    Args:
        d (dict) : Description of :class:`.Variable`.

    Returns:
        item (tuple(str, Variable)) : Tuple containing id (str) and type of :class:`.Variable`.
    '''
    type = d['type'] if isinstance(d['type'], str) else d['type']['value']
    return d['id'], VTYPES.get(type, VTYPES.get(None))


def from_entity_descr(d):
    '''Deserialize entity from data description.

    Args:
        d (dict) : Description of :class:`.Entity`.

    Returns:
        k (dict) : Keyword arguments to instantiate :class:`.Entity` from dataframe.
    '''
    keys = ['index', 'time_index']
    k = dict(zip(keys, map(d.get, keys)))
    k.update(secondary_time_index=d['properties']['secondary_time_index'])
    k.update(variable_types=dict(map(from_var_descr, d['variables'])))
    return k


def read_entity_data(info, path):
    '''Read entity data from disk.

    Args:
        info (dict) : Information for loading entity data.
        path (str) : Location on disk to read entity data.

    Returns:
        df (DataFrame) : Instance of entity dataframe.
    '''
    file = os.path.join(path, info['location'])
    attr = 'read_{}'.format(info['type'])
    return getattr(pd, attr)(file, **info.get('params', {}))


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
