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
_error = '"{}" is not supported'.format


def to_entity_descr(e):
    '''serialize entity to data description'''
    descr = {}
    for key in SCHEMA.get('entities'):
        if key in ['id', 'index', 'time_index']:
            descr[key] = getattr(e, key)
        elif key == 'properties':
            descr[key] = dict(secondary_time_index=e.secondary_time_index)
        elif key == 'variables':
            descr[key] = [v.create_data_description() for v in e.variables]
        elif key == 'loading_info':
            descr[key] = {
                'properties': {
                    'last_time_index': e.last_time_index is not None,
                    'dtypes': e.df.dtypes.astype(str).to_dict(),
                },
                'params': {},
            }
        else:
            raise ValueError(_error(key))
    return e.id, descr


def to_relation_descr(r):
    '''serialize relationship of entity set to data description'''
    return {
        key: [getattr(r, attr).id for attr in attrs]
        for key, attrs in SCHEMA.get('relationships').items()
    }


def write_data(e, path, type='csv', **params):
    '''serialize entity data to disk'''
    basename = '.'.join([e.id, type])
    if 'compression' in params:
        basename += '.' + params['compression']
    location = os.path.join('data', basename)
    loading_info = dict(location=location, type=type.lower())
    assert loading_info['type'] in TYPES, _error(type)
    attr = 'to_{}'.format(loading_info['type'])
    file = os.path.join(path, location)
    obj = e.df.select_dtypes('object').columns
    e.df[obj] = e.df[obj].astype('unicode')
    e.df.columns = e.df.columns.astype('unicode')
    getattr(e.df, attr)(file, **params)
    return loading_info


def write(es, path, **params):
    '''serialize entity set to disk'''
    path = os.path.abspath(path)
    if os.path.exists(path):
        shutil.rmtree(path)
    for p in [path, os.path.join(path, 'data')]:
        os.makedirs(p)
    d = es.create_data_description()
    for e in es.entities:
        info = write_data(e, path, **params)
        d['entities'][e.id]['loading_info'].update(info)
    file = os.path.join(path, 'data_description.json')
    with open(file, 'w') as f:
        json.dump(d, f)


def from_var_descr(v):
    '''deserialize entity variable from data description'''
    type = v['type'] if isinstance(v['type'], str) else v['type']['value']
    return v['id'], VTYPES.get(type, VTYPES.get(None))


def from_entity_descr(e):
    '''deserialize entity from data description'''
    keys = ['index', 'time_index']
    d = dict(zip(keys, map(e.get, keys)))
    d.update(secondary_time_index=e['properties']['secondary_time_index'])
    d.update(variable_types=dict(map(from_var_descr, e['variables'])))
    return d


def read_entity_data(d, path=None, **params):
    '''deserialize entity data from disk or memory'''
    columns = list(d['variable_types'])
    get = params['properties']['dtypes'].get
    dtypes = dict(zip(columns, map(get, columns)))
    if path is None:
        df = pd.DataFrame(columns=columns)
    else:
        file = os.path.join(path, params['location'])
        attr = 'read_{}'.format(params['type'])
        df = getattr(pd, attr)(file, **params.get('params', {}))
    return df.astype(dtypes)


def read_data_description(path):
    '''Deserialize entity set from `data_description.json`.

        Args:
            path (str): Location of root directory to read `data_description.json`.
    '''
    path = os.path.abspath(path)
    assert os.path.exists(path), '"{}" does not exist'.format(path)
    file = os.path.join(path, 'data_description.json')
    with open(file, 'r') as f:
        d = json.load(f)
    d['root'] = path
    return d
