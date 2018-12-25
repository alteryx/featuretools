import os
import json
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
    '''serialize entity of entity set to data description'''
    descr = {}
    for key in SCHEMA.get('entities'):
        if key in ['id', 'index', 'time_index']:
            descr[key] = getattr(e, key)
        elif key == 'properties':
            descr[key] = dict(secondary_time_index=e.secondary_time_index)
        elif key == 'variables':
            descr[key] = [v.create_metadata_dict() for v in e.variables]
        elif key == 'loading_info':
            descr[key] = dict(dtypes=e.df.dtypes.astype(str).to_dict())
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
    '''serialize data of entity to disk'''
    basename = '.'.join([e.id, type])
    if 'compression' in params:
        basename += '.' + params['compression']
    location = os.path.join('data', basename)
    loading_info = dict(location=location, type=type.lower())
    assert loading_info['type'] in TYPES, _error(type)
    attr = 'to_{}'.format(loading_info['type'])
    file = os.path.join(path, location)
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
    '''serialize variable of entity from data description'''
    return v['id'], VTYPES.get(v['dtype_repr'], VTYPES.get(None))


def from_entity_descr(e):
    '''serialize entity from data description'''
    keys = ['index', 'time_index']
    d = dict(zip(keys, map(e.get, keys)))
    d.update(secondary_time_index=e['properties']['secondary_time_index'])
    d.update(variable_types=dict(map(from_var_descr, e['variables'])))
    return d


def read_data(d, path=None, **params):
    '''serialize data of entity from disk or memory'''
    if path is None:
        df = pd.DataFrame(columns=list(d['variable_types']))
    else:
        file = os.path.join(path, params['location'])
        attr = 'read_{}'.format(params['type'])
        df = getattr(pd, attr)(file, **params.get('params', {}))
    return df.astype(params['dtypes'])


def read(path):
    '''serialize entity set from disk'''
    path = os.path.abspath(path)
    assert os.path.exists(path), '"{}" does not exist'.format(path)
    file = os.path.join(path, 'data_description.json')
    with open(file, 'r') as f:
        d = json.load(f)
    d['root'] = path
    return d
